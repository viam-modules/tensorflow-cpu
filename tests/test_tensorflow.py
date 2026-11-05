from src.tensorflow_module import SignatureDefModel, TensorflowModule
from typing import Mapping, Any, Dict
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.services.mlmodel import Metadata
import pytest
import keras
import tensorflow as tf
import numpy as np
from numpy.typing import NDArray


def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
        struct = Struct()
        struct.update(dictionary=dictionary)
        return ComponentConfig(attributes=struct)



class TestTensorflowCPU:

    empty_config = make_component_config({})
    badconfig =make_component_config({
        "model_path": "testModel"
    })
    saved_model_config =make_component_config({
        "model_path": "./tests/EffNet",
        "label_path": "put/Labels/here.txt"
    })
    keras_config = make_component_config({
        "model_path": "./tests/mock_model.keras",
        "label_path": "put/Labels/here.txt"
    })

        
    def getTFCPU(self):
        tfmodel = TensorflowModule("test")
        tfmodel.model = tf.saved_model.load("./tests/EffNet")
        return tfmodel
    
    def getKerasModel(self):
        tfmodel = TensorflowModule("test")
        tfmodel.model = tf.keras.models.load_model("./tests/mock_model.keras")
        tfmodel.is_keras = True
        return tfmodel
        

    def test_validate(self):
        tfm = TensorflowModule("test")
        with pytest.raises(Exception):
            response = tfm.validate_config(config=self.empty_config)
        with pytest.raises(Exception):
            response = tfm.validate_config(config=self.badconfig)
        response = tfm.validate_config(config=self.saved_model_config)
        response = tfm.validate_config(config=self.keras_config)


    @pytest.mark.asyncio
    async def test_saved_model_infer(self):
        tfmodel = self.getTFCPU()
        tfmodel.reconfigure(config=self.saved_model_config, dependencies=None)
        fakeInput = {"input": np.ones([1,10,10,3])}   # make a fake input thingy
        out = await tfmodel.infer(input_tensors=fakeInput) 
        assert isinstance(out, Dict)
        for output in out:
            assert isinstance(out[output], np.ndarray)
 

    @pytest.mark.asyncio
    async def test_saved_model_metadata(self):
        tfmodel = self.getTFCPU()
        tfmodel.reconfigure(config=self.saved_model_config, dependencies=None)
        md = await tfmodel.metadata()
        assert isinstance(md, Metadata)
        assert hasattr(md, "name")
        assert hasattr(md, "input_info")
        assert hasattr(md, "output_info")

    @pytest.mark.asyncio
    async def test_keras_infer(self):
        tfmodel = self.getKerasModel()
        tfmodel.reconfigure(config=self.keras_config, dependencies=None)
        
        fakeInput = {"input": np.ones([1, 4]).astype(np.float32)}
        
        out = await tfmodel.infer(input_tensors=fakeInput)
        assert isinstance(out, Dict)
        for output in out:
            assert isinstance(out[output], np.ndarray)

    @pytest.mark.asyncio
    async def test_keras_metadata(self):
        tfmodel = self.getKerasModel()
        tfmodel.reconfigure(config=self.keras_config, dependencies=None)
        md = await tfmodel.metadata()
        assert isinstance(md, Metadata)
        assert hasattr(md, "name")
        assert hasattr(md, "input_info")
        assert hasattr(md, "output_info")


def make_v1_optimizer_saved_model(path, output_names):
    """Save a SavedModel whose optimizer slots tf.saved_model.load can't rebuild, like Keras 2 legacy optimizers."""
    class Model(tf.Module):
        def __init__(self):
            super().__init__()
            self.w = tf.Variable([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.opt = tf.compat.v1.train.AdamOptimizer()

        @tf.function(input_signature=[tf.TensorSpec([1, 3], tf.float32, name="x")])
        def serve(self, x):
            y = tf.matmul(x, self.w)
            return {name: y * (i + 1) for i, name in enumerate(output_names)}

    model = Model()
    model.opt.minimize(lambda: tf.reduce_sum(model.w) * 0.0, var_list=[model.w])
    tf.saved_model.save(model, str(path), signatures={"serving_default": model.serve})
    with pytest.raises(AttributeError, match="add_slot"):
        tf.saved_model.load(str(path))
    return str(path)


class TestKeras3Compatibility:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("output_names", [["scores"], ["boxes", "scores"]])
    async def test_legacy_optimizer_saved_model(self, tmp_path, output_names):
        path = make_v1_optimizer_saved_model(tmp_path / "model", output_names)
        tfmodel = TensorflowModule("test")
        tfmodel.reconfigure(config=make_component_config({"model_path": path}), dependencies=None)

        md = await tfmodel.metadata()
        assert [(i.name, list(i.shape), i.data_type) for i in md.input_info] == [("x", [1, 3], "float32")]
        assert [(o.name, list(o.shape), o.data_type) for o in md.output_info] == [
            (name, [1, 2], "float32") for name in output_names
        ]

        out = await tfmodel.infer(input_tensors={"x": np.ones([1, 3], dtype=np.float32)})
        assert list(out.keys()) == output_names
        for i, name in enumerate(output_names):
            np.testing.assert_allclose(out[name], [[9.0 * (i + 1), 12.0 * (i + 1)]])

        assert isinstance(tfmodel.model, SignatureDefModel)
        await tfmodel.close()
        with pytest.raises(RuntimeError, match="closed Session"):
            await tfmodel.infer(input_tensors={"x": np.ones([1, 3], dtype=np.float32)})

    @pytest.mark.asyncio
    async def test_keras2_legacy_optimizer_saved_model(self):
        # Keras 2.13 model saved with tf.saved_model.save after fitting with the legacy Adam optimizer (TF 2.13.1, Apple silicon).
        tfmodel = TensorflowModule("test")
        tfmodel.reconfigure(
            config=make_component_config({"model_path": "./tests/keras2_legacy_optimizer"}), dependencies=None
        )
        md = await tfmodel.metadata()
        assert [o.name for o in md.output_info] == ["dense"]

        out = await tfmodel.infer(input_tensors={"image": np.ones([1, 8, 8, 3], dtype=np.uint8)})
        np.testing.assert_allclose(out["dense"], [[0.33564496, 0.3318222, 0.33253285]], rtol=1e-5)
        await tfmodel.close()

    def test_unrelated_attribute_error_is_not_swallowed(self, monkeypatch):
        def fail(_):
            raise AttributeError("something else")

        monkeypatch.setattr(tf.saved_model, "load", fail)
        tfmodel = TensorflowModule("test")
        with pytest.raises(AttributeError, match="something else"):
            tfmodel.reconfigure(config=make_component_config({"model_path": "./tests/EffNet"}), dependencies=None)
        assert tfmodel.model is None

    @pytest.mark.asyncio
    async def test_keras3_saved_model(self, tmp_path):
        inp = keras.Input(shape=(3,), batch_size=1, name="x")
        model = keras.Model(inp, keras.layers.Dense(2, activation="softmax")(inp))
        model.export(str(tmp_path / "model"), verbose=False)

        tfmodel = TensorflowModule("test")
        tfmodel.reconfigure(config=make_component_config({"model_path": str(tmp_path / "model")}), dependencies=None)
        assert not callable(tfmodel.model)

        md = await tfmodel.metadata()
        x = np.ones([1, 3], dtype=np.float32)
        out = await tfmodel.infer(input_tensors={"x": x})
        assert list(out.keys()) == [md.output_info[0].name]
        np.testing.assert_allclose(out[md.output_info[0].name], model.predict(x, verbose=0), rtol=1e-5)