# tensorflow-cpu

Viam provides a `tensorflow-cpu` model of the [ML model service](https://docs.viam.com/ml/deploy/) that allows CPU-based inference on a Tensorflow model in the SavedModel format.

Configure this ML model service as a [modular resource](https://docs.viam.com/modular-resources/) on your robot to take advantage of Tensorflow on the Viam platform--including previously existing or even user-trained models.

## Getting started

The first step is to prepare a valid Tensorflow model.  A valid Tensorflow model comes as a directory which can be named anything.  Within the model directory, there should at least be a `saved_model.pb` file and an internal directory named `variables`, which itself should contain two files: `variables.index` and `variables.data-00000-of-00001`.  The model directory may also include other files (such as `keras_metadata.pb`), but those are irrelevant for now.  The path to the model directory will be important later.


> [!NOTE]  
> Before adding or configuring your module, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

## Configuration

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `mlmodel` type, then select the `tensorflow-cpu` model. Enter a name for your service and click **Create**.

### Example Configuration

```json
{
  "modules": [
    {
      "type": "registry",
      "name": "viam_tensorflow-cpu",
      "module_id": "viam:tensorflow-cpu",
      "version": "latest"
    }
  ],
  "services": [
    {
      "model": "viam:mlmodel:tensorflow-cpu",
      "attributes": {
        "package_reference": null,
        "model_path": "/home/kj/Resnet50/",
        "label_path": "/home/kj/imagenetlabels.txt"
      },
      "name": "myTFModel",
      "type": "mlmodel",
      "namespace": "rdk"
    },
  ]
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam:mlmodel:tensorflow-cpu` services:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_path` | string | **Required** | The full path (on robot) to a valid Tensorflow model directory. |
| `label_path` | string | **Optional** | The full path (on robot) to a text file with class labels |

### Usage

This module is made for use with the following methods of the [ML model service API](https://docs.viam.com/ml/deploy/#api): 
- [`Metadata()`](https://docs.viam.com/ml/deploy/#metadata)
- [`Infer()`](https://docs.viam.com/ml/deploy/#infer)

A call to `Metadata()` will return relevant information about the shape, type, and size of the input and output tensors.  For the `Infer()` method, the module will accept a struct of numpy arrays representing input tensors. The number and dimensionality of the input tensors depends on the included Tensorflow model. It will return a struct of numpy arrays representing output tensors.





