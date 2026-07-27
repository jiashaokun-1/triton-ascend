#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OperationSupport.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <charconv>
#include <limits>
#include <optional>
#include <string>

namespace py = pybind11;

namespace mlir::triton::ascend::ub {
namespace {

std::string requireString(const py::dict &mapping, const char *key) {
  if (!mapping.contains(key))
    throw py::key_error(key);
  return py::cast<std::string>(mapping[key]);
}

PipelineIdentity parsePipelineIdentity(const py::handle &value,
                                       StringRef targetArch) {
  PipelineIdentity identity;
  identity.targetArch = targetArch.str();
  if (py::isinstance<py::str>(value)) {
    identity.sha256 = py::cast<std::string>(value);
    return identity;
  }

  py::dict mapping = py::cast<py::dict>(value);
  identity.openSourcePipeline =
      requireString(mapping, "open_source_pipeline");
  identity.canonicalTtirSha256 =
      requireString(mapping, "canonical_ttir_sha256");
  identity.relevantOptionsJson =
      requireString(mapping, "relevant_options_json");
  identity.targetArch = requireString(mapping, "target_arch");
  identity.tritonVersion = requireString(mapping, "triton_version");
  identity.cannVersionHash = requireString(mapping, "cann_version_hash");
  identity.sha256 = requireString(mapping, "sha256");
  return identity;
}

SmallVector<PipelineStageContext> parsePipelineStages(const py::handle &value) {
  SmallVector<PipelineStageContext> stages;
  for (const py::handle item : py::cast<py::list>(value)) {
    py::dict mapping = py::cast<py::dict>(item);
    PipelineStageContext stage;
    stage.stageName = requireString(mapping, "stage_name");
    if (mapping.contains("options")) {
      py::dict stageOptions = py::cast<py::dict>(mapping["options"]);
      for (auto option : stageOptions) {
        stage.options[py::cast<std::string>(option.first)] =
            py::cast<std::string>(option.second);
      }
    }
    stages.push_back(std::move(stage));
  }
  return stages;
}

bool sameIdentity(const PipelineIdentity &lhs, const PipelineIdentity &rhs) {
  return lhs.openSourcePipeline == rhs.openSourcePipeline &&
         lhs.canonicalTtirSha256 == rhs.canonicalTtirSha256 &&
         lhs.relevantOptionsJson == rhs.relevantOptionsJson &&
         lhs.targetArch == rhs.targetArch &&
         lhs.tritonVersion == rhs.tritonVersion &&
         lhs.cannVersionHash == rhs.cannVersionHash &&
         lhs.sha256 == rhs.sha256;
}

std::optional<PipelineContractBinding>
parseProfileBinding(const py::handle &value) {
  try {
    if (!py::isinstance<py::dict>(value))
      return std::nullopt;
    py::dict mapping = py::cast<py::dict>(value);
    if (!mapping.contains("stage_name") || !mapping.contains("options") ||
        !mapping.contains("contract_id") ||
        !mapping.contains("contract_version"))
      return std::nullopt;
    py::list oneStage;
    py::dict stage;
    stage["stage_name"] = mapping["stage_name"];
    stage["options"] = mapping["options"];
    oneStage.append(std::move(stage));
    SmallVector<PipelineStageContext> parsedStages =
        parsePipelineStages(oneStage);
    PipelineContractBinding binding;
    binding.stage = std::move(parsedStages.front());
    binding.contractId = requireString(mapping, "contract_id");
    binding.contractVersion = requireString(mapping, "contract_version");
    if (mapping.contains("contract_parameters")) {
      py::dict parameters = py::cast<py::dict>(mapping["contract_parameters"]);
      for (auto parameter : parameters)
        binding.parameters[py::cast<std::string>(parameter.first)] =
            py::cast<std::string>(parameter.second);
    }
    return binding;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

bool hasExactParameters(const PipelineContractBinding &binding,
                        ArrayRef<StringRef> names) {
  if (binding.parameters.size() != names.size())
    return false;
  return llvm::all_of(names, [&](StringRef name) {
    return binding.parameters.contains(name);
  });
}

std::optional<int64_t> getPositiveInt64Parameter(
    const PipelineContractBinding &binding, StringRef name) {
  auto found = binding.parameters.find(name);
  if (found == binding.parameters.end() || found->second.empty())
    return std::nullopt;
  int64_t value = 0;
  const char *first = found->second.data();
  const char *last = first + found->second.size();
  auto parsed = std::from_chars(first, last, value);
  if (parsed.ec != std::errc() || parsed.ptr != last || value <= 0)
    return std::nullopt;
  return value;
}

std::unique_ptr<UBResourceContract>
makeProfileContract(const PipelineContractBinding &binding) {
  constexpr StringLiteral alignmentSuffix = "+ub-alignment";
  StringRef contractId = binding.contractId;
  if (binding.contractVersion == "1" &&
      contractId.ends_with(alignmentSuffix)) {
    StringRef baseContractId = contractId.drop_back(alignmentSuffix.size());
    if (baseContractId.empty() ||
        baseContractId.ends_with(alignmentSuffix) ||
        !binding.parameters.contains("expected_resource_count") ||
        !binding.parameters.contains("alignment_bytes"))
      return nullptr;

    PipelineContractBinding baseBinding = binding;
    baseBinding.contractId = baseContractId.str();
    baseBinding.parameters.erase("alignment_bytes");
    std::unique_ptr<UBResourceContract> baseContract =
        makeProfileContract(baseBinding);
    if (!baseContract)
      return nullptr;

    PipelineContractBinding alignmentBinding;
    alignmentBinding.stage = binding.stage;
    alignmentBinding.contractId = "ub-alignment";
    alignmentBinding.contractVersion = "1";
    alignmentBinding.parameters["expected_resource_count"] =
        binding.parameters.at("expected_resource_count");
    alignmentBinding.parameters["alignment_bytes"] =
        binding.parameters.at("alignment_bytes");
    std::unique_ptr<UBResourceContract> alignmentContract =
        makeProfileContract(alignmentBinding);
    if (!alignmentContract)
      return nullptr;

    std::vector<std::unique_ptr<UBResourceContract>> contracts;
    contracts.push_back(std::move(baseContract));
    contracts.push_back(std::move(alignmentContract));
    return makeSequentialContract(binding.contractId, "1", binding.stage,
                                  std::move(contracts));
  }
  if (binding.contractId == "invalidate-unmodeled-stage" &&
      binding.contractVersion == "1" && binding.parameters.empty())
    return makeInvalidateContract(binding.stage);
  if (binding.contractId == "ub-alignment" &&
      binding.contractVersion == "1") {
    constexpr std::array<StringRef, 2> names = {
        "expected_resource_count", "alignment_bytes"};
    if (!hasExactParameters(binding, names))
      return nullptr;
    auto resourceCount =
        getPositiveInt64Parameter(binding, "expected_resource_count");
    auto alignmentBytes =
        getPositiveInt64Parameter(binding, "alignment_bytes");
    if (!resourceCount || !alignmentBytes)
      return nullptr;
    return makeUBAlignmentContract(
        binding.stage, *resourceCount, *alignmentBytes);
  }
  const bool dynamicSourcePreserve =
      binding.contractId == "dynamic-cv-source-preserve" &&
      binding.contractVersion == "1";
  const bool dynamicReplay = binding.contractId == "dynamic-cv-replay" &&
                             binding.contractVersion == "1";
  const bool dynamicResultPreserve =
      binding.contractId == "dynamic-cv-result-preserve" &&
      binding.contractVersion == "1";
  if (dynamicSourcePreserve || dynamicReplay || dynamicResultPreserve) {
    constexpr std::array<StringRef, 7> names = {
        "expected_resource_count", "expected_output_elements",
        "expected_element_bit_width", "expected_source_payload_bytes",
        "projected_payload_bytes", "fixpipe_min_instances",
        "vector_min_instances"};
    const bool validStage =
        (dynamicReplay &&
         binding.stage.stageName == "ttir.dynamic-cv-pipeline") ||
        (dynamicResultPreserve &&
         binding.stage.stageName == "bisheng.ub-affecting-suffix") ||
        (dynamicSourcePreserve &&
         binding.stage.stageName != "ttir.dynamic-cv-pipeline" &&
         binding.stage.stageName != "bisheng.ub-affecting-suffix");
    if (!validStage || !hasExactParameters(binding, names))
      return nullptr;
    auto resourceCount =
        getPositiveInt64Parameter(binding, "expected_resource_count");
    auto outputElements =
        getPositiveInt64Parameter(binding, "expected_output_elements");
    auto elementBitWidth =
        getPositiveInt64Parameter(binding, "expected_element_bit_width");
    auto sourcePayload =
        getPositiveInt64Parameter(binding, "expected_source_payload_bytes");
    auto projectedPayload =
        getPositiveInt64Parameter(binding, "projected_payload_bytes");
    auto fixpipeInstances =
        getPositiveInt64Parameter(binding, "fixpipe_min_instances");
    auto vectorInstances =
        getPositiveInt64Parameter(binding, "vector_min_instances");
    if (!resourceCount || !outputElements || !elementBitWidth ||
        !sourcePayload || !projectedPayload || !fixpipeInstances ||
        !vectorInstances ||
        *elementBitWidth > std::numeric_limits<unsigned>::max())
      return nullptr;
    if (dynamicReplay)
      return makeDynamicCVReplayContract(
          binding.stage, *resourceCount, *outputElements,
          static_cast<unsigned>(*elementBitWidth), *sourcePayload,
          *projectedPayload, *fixpipeInstances, *vectorInstances);
    if (dynamicResultPreserve)
      return makeDynamicCVResultPreserveContract(
          binding.stage, *resourceCount, *outputElements,
          static_cast<unsigned>(*elementBitWidth), *sourcePayload,
          *projectedPayload, *fixpipeInstances, *vectorInstances);
    return makeDynamicCVSourcePreserveContract(
        binding.stage, *resourceCount, *outputElements,
        static_cast<unsigned>(*elementBitWidth), *sourcePayload,
        *projectedPayload, *fixpipeInstances, *vectorInstances);
  }
  // Dynamic CV changes core projection, dataflow, liveness and cache
  // multiplicity. Reusing an unrelated family-level Preserve/Transform
  // contract here would incorrectly treat those changes as no-ops. This exact
  // stage may therefore carry only the explicit fail-closed contract or the
  // replay-backed dynamic contract handled above.
  if (binding.stage.stageName == "ttir.dynamic-cv-pipeline")
    return nullptr;

  constexpr std::array<StringRef, 4> resourceParameters = {
      "expected_resource_count", "expected_source_elements",
      "expected_element_bit_width", "expected_input_payload_bytes"};
  const bool preserve = binding.contractId == "direct-copy-preserve" &&
                        binding.contractVersion == "1";
  const bool transform = binding.contractId == "direct-copy-max-tiles" &&
                         binding.contractVersion == "1";
  const bool binaryPreserve = binding.contractId == "binary-add-preserve" &&
                              binding.contractVersion == "1";
  const bool binaryTransform =
      binding.contractId == "binary-add-max-tiles" &&
      binding.contractVersion == "1";
  const bool loopPreserve =
      binding.contractId == "loop-carried-add-preserve" &&
      binding.contractVersion == "1";
  const bool loopTransform =
      binding.contractId == "loop-carried-add-max-tiles" &&
      binding.contractVersion == "1";
  const bool loopMultiBuffer =
      binding.contractId == "loop-carried-add-multibuffer" &&
      binding.contractVersion == "1";
  const bool reshapePreserve =
      binding.contractId == "reshape-copy-preserve" &&
      binding.contractVersion == "1";
  const bool reshapeTransform =
      binding.contractId == "reshape-copy-max-tiles" &&
      binding.contractVersion == "1";
  const bool reductionPreserve =
      binding.contractId == "reduction-sum-preserve" &&
      binding.contractVersion == "1";
  const bool reductionTransform =
      binding.contractId == "reduction-sum-max-tiles" &&
      binding.contractVersion == "1";
  const bool reductionExtraBuffer =
      binding.contractId == "reduction-sum-extra-buffer" &&
      binding.contractVersion == "1";
  SmallVector<StringRef> expectedNames(resourceParameters.begin(),
                                       resourceParameters.end());
  if (reductionPreserve || reductionTransform || reductionExtraBuffer) {
    expectedNames.push_back("expected_scratch_payload_bytes");
    expectedNames.push_back("expected_accumulator_payload_bytes");
  }
  if (transform || binaryTransform || loopTransform || reshapeTransform ||
      reductionTransform)
    expectedNames.push_back("max_tiles");
  if (loopMultiBuffer)
    expectedNames.push_back("expected_step_input_instances");
  if ((!preserve && !transform && !binaryPreserve && !binaryTransform &&
       !loopPreserve && !loopTransform && !loopMultiBuffer &&
       !reshapePreserve &&
       !reshapeTransform && !reductionPreserve && !reductionTransform &&
       !reductionExtraBuffer) ||
      !hasExactParameters(binding, expectedNames))
    return nullptr;

  auto resourceCount =
      getPositiveInt64Parameter(binding, "expected_resource_count");
  auto sourceElements =
      getPositiveInt64Parameter(binding, "expected_source_elements");
  auto elementBitWidth =
      getPositiveInt64Parameter(binding, "expected_element_bit_width");
  auto inputPayload =
      getPositiveInt64Parameter(binding, "expected_input_payload_bytes");
  if (!resourceCount || !sourceElements || !elementBitWidth ||
      !inputPayload ||
      *elementBitWidth > std::numeric_limits<unsigned>::max())
    return nullptr;
  std::optional<int64_t> scratchPayload;
  std::optional<int64_t> accumulatorPayload;
  if (reductionPreserve || reductionTransform || reductionExtraBuffer) {
    scratchPayload = getPositiveInt64Parameter(
        binding, "expected_scratch_payload_bytes");
    accumulatorPayload = getPositiveInt64Parameter(
        binding, "expected_accumulator_payload_bytes");
    if (!scratchPayload || !accumulatorPayload)
      return nullptr;
  }
  if (preserve)
    return makeDirectCopyPreserveContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload);
  if (binaryPreserve)
    return makeBinaryAddPreserveContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload);
  if (loopPreserve)
    return makeLoopCarriedAddPreserveContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload);
  if (reshapePreserve)
    return makeReshapeCopyPreserveContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload);
  if (reductionPreserve)
    return makeReductionSumPreserveContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload,
        *scratchPayload, *accumulatorPayload);
  if (reductionExtraBuffer)
    return makeReductionSumExtraBufferContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload,
        *scratchPayload, *accumulatorPayload);
  if (loopMultiBuffer) {
    auto stepInputInstances = getPositiveInt64Parameter(
        binding, "expected_step_input_instances");
    if (!stepInputInstances)
      return nullptr;
    return makeLoopCarriedAddMultiBufferContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload,
        *stepInputInstances);
  }

  auto maxTiles = getPositiveInt64Parameter(binding, "max_tiles");
  if (!maxTiles)
    return nullptr;
  if (binaryTransform)
    return makeBinaryAddMaxTilesContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload, *maxTiles);
  if (loopTransform)
    return makeLoopCarriedAddMaxTilesContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload, *maxTiles);
  if (reshapeTransform)
    return makeReshapeCopyMaxTilesContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload, *maxTiles);
  if (reductionTransform)
    return makeReductionSumMaxTilesContract(
        binding.stage, *resourceCount, *sourceElements,
        static_cast<unsigned>(*elementBitWidth), *inputPayload,
        *scratchPayload, *accumulatorPayload, *maxTiles);
  return makeDirectCopyMaxTilesContract(
      binding.stage, *resourceCount, *sourceElements,
      static_cast<unsigned>(*elementBitWidth), *inputPayload, *maxTiles);
}

bool loadMatchingProfile(const py::handle &value,
                         const PipelineIdentity &identity,
                         ArrayRef<PipelineStageContext> actualStages,
                         PipelineContractRegistry &registry,
                         bool allowUncertifiedActiveContracts) {
  py::dict document = py::cast<py::dict>(value);
  py::list profiles = py::cast<py::list>(document["profiles"]);
  std::optional<SmallVector<PipelineContractBinding>> matchedBindings;
  bool matchedProfileIsCertified = false;
  for (const py::handle item : profiles) {
    try {
      if (!py::isinstance<py::dict>(item))
        continue;
      py::dict profile = py::cast<py::dict>(item);
      if (!profile.contains("pipeline_identity") ||
          !profile.contains("pipeline_stages") ||
          !py::isinstance<py::list>(profile["pipeline_stages"]))
        continue;
      PipelineIdentity profileIdentity = parsePipelineIdentity(
          profile["pipeline_identity"], identity.targetArch);
      if (!sameIdentity(profileIdentity, identity))
        continue;
      if (matchedBindings)
        return false;

      SmallVector<PipelineContractBinding> bindings;
      for (const py::handle rawBinding :
           py::cast<py::list>(profile["pipeline_stages"])) {
        std::optional<PipelineContractBinding> binding =
            parseProfileBinding(rawBinding);
        if (!binding)
          return false;
        bindings.push_back(std::move(*binding));
      }
      matchedBindings = std::move(bindings);
      try {
        if (profile.contains("contract_version") &&
            requireString(profile, "contract_version") == "ttir-ub-lb-v1" &&
            profile.contains("oracle_report_sha256") &&
            profile.contains("semantic_model_sha256") &&
            profile.contains("validated_seeds") &&
            profile.contains("retry_validated") &&
            profile.contains("auto_tile_and_bind_subblock_outcome")) {
          const std::string reportHash =
              requireString(profile, "oracle_report_sha256");
          const std::string semanticModelHash =
              requireString(profile, "semantic_model_sha256");
          py::list seeds = py::cast<py::list>(profile["validated_seeds"]);
          const py::handle autoTileOutcome =
              profile["auto_tile_and_bind_subblock_outcome"];
          bool validHash = reportHash.size() == 64 &&
                           llvm::all_of(reportHash, [](char character) {
                             return (character >= '0' && character <= '9') ||
                                    (character >= 'a' && character <= 'f');
                           });
          bool validSemanticModelHash =
              semanticModelHash.size() == 64 &&
              llvm::all_of(semanticModelHash, [](char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f');
              });
          bool validSeeds = seeds.size() == 20;
          for (size_t ordinal = 0; validSeeds && ordinal < seeds.size();
               ++ordinal) {
            py::handle seed = seeds[ordinal];
            validSeeds = py::isinstance<py::int_>(seed) &&
                         !py::isinstance<py::bool_>(seed) &&
                         py::cast<int64_t>(seed) ==
                             static_cast<int64_t>(ordinal);
          }
          const py::handle retry = profile["retry_validated"];
          matchedProfileIsCertified =
              validHash && validSemanticModelHash && validSeeds &&
              py::isinstance<py::bool_>(retry) &&
              py::cast<bool>(retry) &&
              py::isinstance<py::bool_>(autoTileOutcome);
        }
      } catch (const std::exception &) {
        matchedProfileIsCertified = false;
      }
    } catch (const std::exception &) {
      continue;
    }
  }
  if (!matchedBindings || matchedBindings->size() != actualStages.size())
    return false;

  const auto isFamilyMaterializationContract =
      [](const PipelineContractBinding &binding) {
    StringRef contractId = binding.contractId;
    constexpr StringLiteral alignmentSuffix = "+ub-alignment";
    if (contractId.ends_with(alignmentSuffix))
      contractId = contractId.drop_back(alignmentSuffix.size());
    return contractId == "direct-copy-preserve" ||
           contractId == "direct-copy-max-tiles" ||
           contractId == "binary-add-preserve" ||
           contractId == "binary-add-max-tiles" ||
           contractId == "loop-carried-add-preserve" ||
           contractId == "loop-carried-add-max-tiles" ||
           contractId == "loop-carried-add-multibuffer" ||
           contractId == "reshape-copy-preserve" ||
           contractId == "reshape-copy-max-tiles" ||
           contractId == "reduction-sum-preserve" ||
           contractId == "reduction-sum-max-tiles" ||
           contractId == "reduction-sum-extra-buffer" ||
           contractId == "dynamic-cv-source-preserve" ||
           contractId == "dynamic-cv-replay" ||
           contractId == "dynamic-cv-result-preserve";
  };
  const bool containsFamilyMaterializationContract =
      llvm::any_of(*matchedBindings, isFamilyMaterializationContract);
  const bool containsAlignmentContract = llvm::any_of(
      *matchedBindings, [](const PipelineContractBinding &binding) {
        return binding.contractId == "ub-alignment";
      });
  if (containsAlignmentContract)
    return false;
  const bool containsActiveContract =
      containsFamilyMaterializationContract;
  if (containsActiveContract && !allowUncertifiedActiveContracts &&
      !matchedProfileIsCertified)
    return false;

  registry.setProfileIdentity(identity);
  for (PipelineContractBinding &binding : *matchedBindings) {
    std::unique_ptr<UBResourceContract> contract =
        makeProfileContract(binding);
    if (!contract)
      return false;
    if (failed(registry.addProfileContract(std::move(binding),
                                           std::move(contract))))
      return false;
  }
  return registry.matchesProfile(identity, actualStages);
}

void validatePackagedProfileShape(const py::handle &value) {
  py::dict profile = py::cast<py::dict>(value);
  if (requireString(profile, "schema") != "ttir-ub-lb-profile-v1")
    throw py::value_error("unsupported contract_profile schema");
  if (!profile.contains("profiles") ||
      !py::isinstance<py::list>(profile["profiles"]))
    throw py::type_error("contract_profile.profiles must be a list");
}

TTIRUBAnalysisOptions parseOptions(const py::dict &mapping) {
  static const llvm::StringSet<> allowedKeys = {
      "arch", "compile_mode", "pipeline_identity", "pipeline_stages",
      "contract_profile"};
  for (auto item : mapping) {
    std::string key = py::cast<std::string>(item.first);
    if (!allowedKeys.contains(key))
      throw py::key_error(key);
  }

  TTIRUBAnalysisOptions options;
  options.targetArch = requireString(mapping, "arch");
  options.compileMode = requireString(mapping, "compile_mode");
  if (!mapping.contains("pipeline_identity"))
    throw py::key_error("pipeline_identity");
  options.pipelineIdentity =
      parsePipelineIdentity(mapping["pipeline_identity"], options.targetArch);
  if (!mapping.contains("pipeline_stages"))
    throw py::key_error("pipeline_stages");
  options.stages = parsePipelineStages(mapping["pipeline_stages"]);
  if (!mapping.contains("contract_profile"))
    throw py::key_error("contract_profile");
  validatePackagedProfileShape(mapping["contract_profile"]);
  return options;
}

py::dict serializeCertificate(const LowerBoundCertificate &certificate) {
  py::dict result;
  result["kind"] = certificate.kind;
  result["bytes"] = certificate.bytes;
  py::list resourceIds;
  for (ResourceId id : certificate.resourceIds)
    resourceIds.append(id);
  result["resource_ids"] = std::move(resourceIds);
  py::list contractTrace;
  for (const std::string &contractId : certificate.contractTrace)
    contractTrace.append(contractId);
  result["contract_trace"] = std::move(contractTrace);
  return result;
}

py::dict serializeResult(const TTIRUBAnalysisResult &analysis,
                         const PipelineIdentity &identity) {
  py::dict result;
  result["decision"] =
      analysis.decision == TTIRUBDecision::Reject ? "reject" : "defer";
  result["lower_bound_bytes"] = analysis.lowerBoundBytes;
  if (analysis.capacityBytes)
    result["capacity_bytes"] = *analysis.capacityBytes;
  else
    result["capacity_bytes"] = py::none();

  py::list certificates;
  for (const LowerBoundCertificate &certificate : analysis.certificates)
    certificates.append(serializeCertificate(certificate));
  result["certificates"] = std::move(certificates);

  py::list unsupportedReasons;
  for (const std::string &reason : analysis.unsupportedReasons)
    unsupportedReasons.append(reason);
  result["unsupported_reasons"] = std::move(unsupportedReasons);
  py::list deferTrace;
  for (const std::string &trace : analysis.deferTrace)
    deferTrace.append(trace);
  result["defer_trace"] = std::move(deferTrace);
  result["pipeline_identity"] = identity.sha256;
  result["contract_version"] = analysis.contractVersion;
  return result;
}

py::dict runAnalysis(ModuleOp &module, const py::dict &rawOptions,
                     bool allowUncertifiedActiveContracts) {
  TTIRUBAnalysisOptions options = parseOptions(rawOptions);
  PipelineContractRegistry registry;
  loadMatchingProfile(rawOptions["contract_profile"],
                      options.pipelineIdentity, options.stages, registry,
                      allowUncertifiedActiveContracts);
  TTIRUBAnalysisResult analysis =
      analyzeTTIRUBLowerBound(module, options, registry);
  return serializeResult(analysis, options.pipelineIdentity);
}

} // namespace

void initTTIRUBLowerBoundBindings(py::module_ &module) {
  module.def("ttir_ub_portable_module_text", [](ModuleOp &module) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module.print(stream, OpPrintingFlags().enableDebugInfo(false));
    stream.flush();
    return text;
  });

  module.def("ttir_ub_generic_module_text", [](ModuleOp &module) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module.print(stream, OpPrintingFlags().printGenericOpForm());
    stream.flush();
    return text;
  });

  module.def("get_ub_capacity_bytes", [](const std::string &arch) -> py::object {
    std::optional<int64_t> capacity = getUBCapacityBytes(arch);
    if (!capacity)
      return py::none();
    return py::int_(*capacity);
  });

  module.def("ttir_ub_lower_bound",
             [](ModuleOp &module, const py::dict &rawOptions) {
               return runAnalysis(module, rawOptions, false);
             });
  module.def("ttir_ub_lower_bound_candidate_for_oracle",
             [](ModuleOp &module, const py::dict &rawOptions) {
               return runAnalysis(module, rawOptions, true);
             });
}

} // namespace mlir::triton::ascend::ub
