# The app

Commandline args will be less or more self-explanatory for now to describe the app.

Say hello to our simple program demonstrating how we can detect cats in a video stream in real-time.

```
Usage: is-you-cat [params] 

        --camera (value:0)
                Camera device number.
        --class-treshold (value:0.5)
                What is the probability treshold for being cat (if model outputs positive class (cat) probability greater than class-treshold then input will be reported as cat)
        --desired-fps (value:30)
                Desired camera FPS
        --detect-motion (value:true)
                Whether to stop model running when no motion detected
        --frame-count (value:0)
                After how many frames to exit (0 means run endlessly)
        --frame-timeout (value:-1)
                After how many frames force model run (0 means always, -1 means never)
        --full-screen (value:false)
                Run app in full screen mode
        -h, --help

        --model (value:../models/onnx/MobilenetV3_torch_first.onnx)
                Path to exported model or first part of exported split model
        --second-model (value:../models/onnx/MobilenetV3_torch_second.onnx)
                Second part of split model if --model is split model's first part (optional)
```
