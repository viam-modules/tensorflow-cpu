import asyncio

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.mlmodel import MLModel
from src.tensorflow_module import TensorflowModule



async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """

    Registry.register_resource_creator(
        MLModel.SUBTYPE,
        TensorflowModule.MODEL,
        ResourceCreatorRegistration(TensorflowModule.new_service, TensorflowModule.validate_config))
    module = Module.from_args()

    module.add_model_from_registry(MLModel.SUBTYPE, TensorflowModule.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())