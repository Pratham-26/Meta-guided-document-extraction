import importlib
import sys
import types


def ensure_colbert_compat():
    _patch_langchain_retrievers()
    _patch_colbert_torch_extensions()
    _patch_hf_colbert_tied_weights()


def _patch_langchain_retrievers():
    if "langchain.retrievers" in sys.modules:
        return

    try:
        import langchain.retrievers.document_compressors.base

        return
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        from langchain_classic.retrievers.document_compressors import base as _base

        compressors_mod = types.ModuleType("langchain.retrievers.document_compressors")
        compressors_mod.base = _base
        compressors_mod.BaseDocumentCompressor = _base.BaseDocumentCompressor

        retrievers_mod = types.ModuleType("langchain.retrievers")
        retrievers_mod.document_compressors = compressors_mod

        sys.modules["langchain.retrievers"] = retrievers_mod
        sys.modules["langchain.retrievers.document_compressors"] = compressors_mod
        sys.modules["langchain.retrievers.document_compressors.base"] = _base
    except ImportError:
        pass


def _patch_colbert_torch_extensions():
    try:
        from colbert.modeling.colbert import ColBERT

        if hasattr(ColBERT, "_compat_patched"):
            return

        _orig = ColBERT.try_load_torch_extensions

        def _safe_load(cls, use_gpu):
            try:
                _orig(use_gpu)
            except Exception:
                cls.segmented_maxsim = cls._segmented_maxsim_pure_python
                cls.loaded_extensions = True

        ColBERT.try_load_torch_extensions = classmethod(_safe_load)
        ColBERT._compat_patched = True
    except ImportError:
        pass


def _patch_hf_colbert_tied_weights():
    try:
        import colbert.modeling.hf_colbert as hf_mod

        if getattr(hf_mod, "_compat_patched", False):
            return

        _orig_factory = hf_mod.class_factory

        def _patched_factory(name_or_path):
            cls = _orig_factory(name_or_path)
            if not hasattr(cls, "all_tied_weights_keys"):
                cls.all_tied_weights_keys = {}
            return cls

        hf_mod.class_factory = _patched_factory
        hf_mod._compat_patched = True
    except ImportError:
        pass
