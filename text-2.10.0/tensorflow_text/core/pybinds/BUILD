# Code that exposes C++ libraries to Python via pybind11.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_text:__subpackages__",
])

pybind_extension(
    name = "tflite_registrar",
    srcs = [
        "tflite_registrar.cc",
    ],
    additional_exported_symbols = [
        "AddFastBertNormalize",
        "AddFastSentencepieceTokenize",
        "AddFastSentencepieceDetokenize",
        "AddFastWordpieceDetokenize",
        "AddFastWordpieceTokenize",
        "AddNgramsStringJoin",
        "AddWhitespaceTokenize",
    ],
    module_name = "tflite_registrar",
    deps = [
        "@pybind11",
        # lite:framework tensorflow dep,
        # lite/c:common tensorflow dep,
        # lite/kernels:builtin_ops tensorflow dep,
        "//tensorflow_text/core/kernels:tflite_ops",
    ],
)

pybind_extension(
    name = "pywrap_fast_bert_normalizer_model_builder",
    srcs = ["pywrap_fast_bert_normalizer_model_builder.cc"],
    additional_exported_symbols = [
        "BuildFastBertNormalizerModel",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_fast_bert_normalizer_model_builder",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels:fast_bert_normalizer_model_builder",
        "@pybind11",
    ],
)

py_test(
    name = "pywrap_fast_bert_normalizer_model_builder_test",
    srcs = ["pywrap_fast_bert_normalizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model.fb",
        "//tensorflow_text:python/ops/test_data/fast_bert_normalizer_model_lower_case_nfd_strip_accents.fb",
    ],
    python_version = "PY3",
    deps = [
        ":pywrap_fast_bert_normalizer_model_builder",
        # tensorflow package dep,
    ],
)

pybind_extension(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder.cc"],
    additional_exported_symbols = [
        "BuildFastWordpieceModel",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_fast_wordpiece_tokenizer_model_builder",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels:fast_wordpiece_tokenizer_model_builder",
        "@pybind11",
    ],
)

py_test(
    name = "pywrap_fast_wordpiece_tokenizer_model_builder_test",
    srcs = ["pywrap_fast_wordpiece_tokenizer_model_builder_test.py"],
    data = [
        "//tensorflow_text:python/ops/test_data/fast_wordpiece_tokenizer_model.fb",
    ],
    python_version = "PY3",
    deps = [
        ":pywrap_fast_wordpiece_tokenizer_model_builder",
        # tensorflow package dep,
    ],
)

pybind_extension(
    name = "pywrap_model_converter",
    srcs = ["pywrap_model_converter.cc"],
    additional_exported_symbols = [
        "ConvertSentencepieceModel",
        "ConvertSentencepieceModelForDecoder",
        "GetVocabularySize",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_model_converter",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels/sentencepiece:model_converter",
        "@pybind11",
    ],
)

pybind_extension(
    name = "pywrap_whitespace_tokenizer_config_builder",
    srcs = ["pywrap_whitespace_tokenizer_config_builder.cc"],
    additional_exported_symbols = [
        "BuildWhitespaceTokenizerConfig",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pywrap_whitespace_tokenizer_config_builder",
    srcs_version = "PY3ONLY",
    deps = [
        "//tensorflow_text/core/kernels:whitespace_tokenizer_config_builder",
        "@pybind11",
    ],
)

py_test(
    name = "pywrap_whitespace_tokenizer_config_builder_test",
    srcs = ["pywrap_whitespace_tokenizer_config_builder_test.py"],
    python_version = "PY3",
    deps = [
        ":pywrap_whitespace_tokenizer_config_builder",
        # tensorflow package dep,
    ],
)
