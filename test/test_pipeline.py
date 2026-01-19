import os

def test_model_file_created():
    assert os.path.exists("model/model.pkl")
