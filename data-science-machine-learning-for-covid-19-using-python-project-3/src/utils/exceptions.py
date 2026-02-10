class PipelineError(Exception):
    pass

class DataLoadError(PipelineError):
    pass

class PreprocessingError(PipelineError):
    pass

class ModelError(PipelineError):
    pass
