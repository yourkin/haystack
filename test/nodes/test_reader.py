import os
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from haystack.schema import Document
from haystack.nodes import FARMReader, TransformersReader


@pytest.fixture
def reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            use_gpu=False,
            top_k_per_sample=5,
            no_ans_boost=0,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            tokenizer="deepset/bert-medium-squad2-distilled",
            use_gpu=-1,
            top_k_per_candidate=5,
        )
    raise ValueError(f"Unknown reader: {request.params}")


@pytest.fixture
def no_answer_reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            use_gpu=False,
            top_k_per_sample=5,
            no_ans_boost=0,
            return_no_answer=True,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            tokenizer="deepset/bert-medium-squad2-distilled",
            use_gpu=-1,
            top_k_per_candidate=5,
            return_no_answers=True,
        )
    raise ValueError(f"Unknown reader: {request.params}")


def test_model_download_options():
    # download disabled and model is not cached locally
    with pytest.raises(OSError):
        FARMReader("mfeb/albert-xxlarge-v2-squad2", local_files_only=True, num_processes=0)


def test_farm_reader_invalid_params():
    # invalid max_seq_len (greater than model maximum seq length)
    with pytest.raises(Exception):
        FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, max_seq_len=513)

    # invalid max_seq_len (max_seq_len >= doc_stride)
    with pytest.raises(Exception):
        FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, max_seq_len=129, doc_stride=128)

    # invalid doc_stride (doc_stride >= (max_seq_len - max_query_length))
    with pytest.raises(Exception):
        FARMReader(model_name_or_path="deepset/tinyroberta-squad2", use_gpu=False, doc_stride=999)


@pytest.mark.integration
def test_farm_reader_load_hf_online():
    # Test Case: 1. HuggingFace Hub (online load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    FARMReader(model_name_or_path=hf_model, use_gpu=False, no_ans_boost=0, num_processes=0)


@pytest.mark.integration
def test_farm_reader_load_hf_local(tmp_path):
    # Test Case: 2. HuggingFace downloaded (local load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    local_model_path = tmp_path / "locally_saved_hf"
    model_path = snapshot_download(repo_id=hf_model, revision="main", cache_dir=local_model_path)
    FARMReader(model_name_or_path=model_path, use_gpu=False, no_ans_boost=0, num_processes=0)


@pytest.mark.integration
def test_farm_reader_load_farm_local(tmp_path):
    # Test Case: 3. HF Model saved as FARM Model (same works for trained FARM model) (local load)

    hf_model = "hf-internal-testing/tiny-random-RobertaForQuestionAnswering"
    local_model_path = tmp_path / "locally_saved_farm"
    reader = FARMReader(model_name_or_path=hf_model, use_gpu=False, no_ans_boost=0, num_processes=0)
    reader.save(Path(local_model_path))
    FARMReader(model_name_or_path=local_model_path, use_gpu=False, no_ans_boost=0, num_processes=0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name",
    ["deepset/tinyroberta-squad2", "deepset/bert-medium-squad2-distilled", "deepset/xlm-roberta-base-squad2-distilled"],
)
def test_farm_reader_onnx_conversion_and_inference(model_name, tmpdir, docs):
    FARMReader.convert_to_onnx(model_name=model_name, output_path=Path(tmpdir, "onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "model.onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "processor_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "onnx_model_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "language_model_config.json"))

    reader = FARMReader(str(Path(tmpdir, "onnx")))
    result = reader.predict(query="Where does Paul live?", documents=[docs[0]])
    assert result["answers"][0].answer == "New York"


#
#  MOCK THESE
#


def test_reader_skips_empty_documents(reader):
    predictions, _ = reader.run(query="test query", documents=[Document(content="")])
    assert predictions["answers"] == []


def test_reader_skips_empty_documents_batch(reader):
    predictions, _ = reader.run_batch(
        queries=["test query", "test query"], documents=[[Document(content="")], [Document(content="test answer")]]
    )
    assert predictions["answers"][0] == []
    assert predictions["answers"][1][0].answer != []
