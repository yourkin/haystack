from haystack.nodes import FARMReader


from ..conftest import SAMPLES_PATH


def test_reader_training(tmp_path):
    max_seq_len = 16
    max_query_length = 8
    reader = FARMReader(
        model_name_or_path="deepset/tinyroberta-squad2",
        use_gpu=False,
        num_processes=0,
        max_seq_len=max_seq_len,
        doc_stride=2,
        max_query_length=max_query_length,
    )

    save_dir = f"{tmp_path}/test_dpr_training"
    reader.train(
        data_dir=str(SAMPLES_PATH / "training" / "squad"),
        train_filename="tiny.json",
        dev_filename="tiny.json",
        test_filename="tiny.json",
        n_epochs=1,
        batch_size=1,
        grad_acc_steps=1,
        save_dir=save_dir,
        evaluate_every=2,
        max_seq_len=max_seq_len,
        max_query_length=max_query_length,
    )
