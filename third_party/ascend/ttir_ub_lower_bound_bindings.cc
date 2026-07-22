#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "llvm/ADT/StringSet.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    return binding;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

bool loadMatchingProfile(const py::handle &value,
                         const PipelineIdentity &identity,
                         ArrayRef<PipelineStageContext> actualStages,
                         PipelineContractRegistry &registry) {
  py::dict document = py::cast<py::dict>(value);
  py::list profiles = py::cast<py::list>(document["profiles"]);
  std::optional<SmallVector<PipelineContractBinding>> matchedBindings;
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
    } catch (const std::exception &) {
      continue;
    }
  }
  if (!matchedBindings || matchedBindings->size() != actualStages.size())
    return false;

  registry.setProfileIdentity(identity);
  for (PipelineContractBinding &binding : *matchedBindings) {
    std::unique_ptr<UBResourceContract> contract;
    // P0 deliberately exposes only a fail-closed production constructor.
    // Preserve/Transform constructors must be added here only after their
    // packaged profiles pass the oracle promotion gate.
    if (binding.contractId == "invalidate-unmodeled-stage" &&
        binding.contractVersion == "1")
      contract = makeInvalidateContract(binding.stage);
    else
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
  result["pipeline_identity"] = identity.sha256;
  result["contract_version"] = analysis.contractVersion;
  return result;
}

} // namespace

void initTTIRUBLowerBoundBindings(py::module_ &module) {
  module.def("get_ub_capacity_bytes", [](const std::string &arch) -> py::object {
    std::optional<int64_t> capacity = getUBCapacityBytes(arch);
    if (!capacity)
      return py::none();
    return py::int_(*capacity);
  });

  module.def("ttir_ub_lower_bound",
             [](ModuleOp &module, const py::dict &rawOptions) {
               TTIRUBAnalysisOptions options = parseOptions(rawOptions);
               PipelineContractRegistry registry;
               loadMatchingProfile(rawOptions["contract_profile"],
                                   options.pipelineIdentity, options.stages,
                                   registry);
               TTIRUBAnalysisResult analysis =
                   analyzeTTIRUBLowerBound(module, options, registry);
               return serializeResult(analysis, options.pipelineIdentity);
             });
}

} // namespace mlir::triton::ascend::ub
