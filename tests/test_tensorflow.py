from src.tensorflow_module import TensorflowModule
from typing import Mapping, Any, Dict
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.services.mlmodel import Metadata
import pytest
import tensorflow as tf
import numpy as np
from numpy.typing import NDArray
import keras

def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
        struct = Struct()
        struct.update(dictionary=dictionary)
        return ComponentConfig(attributes=struct)

def make_sequential_keras_model():
    model = keras.Sequential([keras.layers.Dense(10), keras.layers.Dense(5)])
    model.save("./tests/testmodel.keras")

class TestTensorflowCPU:

    empty_config = make_component_config({})
    make_sequential_keras_model()

    badconfig =make_component_config({
        "model_path": "testModel"
    })
    goodconfig =make_component_config({
        "model_path": "./tests/EffNet",
        "label_path": "put/Labels/here.txt"
    })
    config_keras = make_component_config({
        "model_path": "./tests/testmodel.keras",
        "label_path": "put/Labels/here.txt"
    })

        
    def getTFCPU(self):
        tfmodel = TensorflowModule("test")
        tfmodel.model = tf.saved_model.load("./tests/EffNet")
        return tfmodel

    def test_validate(self):
        tfm = TensorflowModule("test")
        with pytest.raises(Exception):
            response = tfm.validate_config(config=self.empty_config)
        with pytest.raises(Exception):
            response = tfm.validate_config(config=self.badconfig)
        response = tfm.validate_config(config=self.goodconfig)
        keras_response = tfm.validate_config(config=self.config_keras)

    @pytest.mark.asyncio
    async def test_infer(self):
        tfmodel = self.getTFCPU()
        tfmodel.reconfigure(config=self.goodconfig, dependencies=None)
        fakeInput = {"input": np.ones([1,10,10,3])}   # make a fake input thingy
        out = await tfmodel.infer(input_tensors=fakeInput)
        assert isinstance(out, Dict)
        for output in out:
            assert isinstance(out[output], np.ndarray)
 

    @pytest.mark.asyncio
    async def test_metadata(self):
        tfmodel = self.getTFCPU()
        tfmodel.reconfigure(config=self.goodconfig, dependencies=None)
        md = await tfmodel.metadata()
        assert isinstance(md, Metadata)
        assert hasattr(md, "name")
        assert hasattr(md, "input_info")
        assert hasattr(md, "output_info")

    # KERAS TESTS
    def getTFCPUKeras(self):
        tfmodel = TensorflowModule("test")
        tfmodel.model = tf.keras.models.load_model("./tests/testmodel.keras")
        return tfmodel
    
    @pytest.mark.asyncio
    async def test_infer_keras(self):
        tf_keras_model = TensorflowModule("test")
        tf_keras_model.reconfigure(config=self.config_keras, dependencies=None)
        fakeInput = {"input_1": np.ones([1, 5])}
        out = await tf_keras_model.infer(input_tensors=fakeInput)
        assert isinstance(out, Dict)
        for output in out:
            assert isinstance(out[output], np.ndarray)

    @pytest.mark.asyncio
    async def test_metadata_keras(self):
        tf_keras_model = self.getTFCPUKeras()
        tf_keras_model.reconfigure(config=self.config_keras, dependencies=None)
        md = await tf_keras_model.metadata()
        assert isinstance(md, Metadata)
        assert hasattr(md, "name")
        assert hasattr(md, "input_info")
        assert hasattr(md, "output_info")
