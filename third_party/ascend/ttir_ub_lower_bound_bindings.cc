#include "Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h"
#include "Analysis/TTIRUBLowerBound/UBResourceContract.h"

#include "llvm/ADT/StringSet.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
               // Production contracts are intentionally unavailable until an
               // oracle-certified packaged profile is implemented. Caller
               // dictionaries are shape-checked but never populate this
               // testing-only registry.
               PipelineContractRegistry registry;
               TTIRUBAnalysisResult analysis =
                   analyzeTTIRUBLowerBound(module, options, registry);
               return serializeResult(analysis, options.pipelineIdentity);
             });
}

} // namespace mlir::triton::ascend::ub
