# DGMN-pytorch

An unofficial repository of the paper 'A Document-grounded Matching Network for Response Selection in Retrieval-based Chatbots', which is implemented in PyTorch.

## Dependencies

+ Python 3.7
+ PyTorch 1.7.0
+ fitlog

## Reproduced Results

<table>
    <tr>
        <th rowspan="3">Model</th>
        <th colspan="3" rowspan="2">CMUDoG</th>
        <th colspan="6">PersonaChat</th>
    </tr>
    <tr>
        <th colspan="3">Original Persona</th>
        <th colspan="3">Revised Persona</th>
    </tr>
    <tr>
        <td>R@1</td>
        <td>R@2</td>
        <td>R@5</td>
        <td>R@1</td>
        <td>R@2</td>
        <td>R@5</td>
        <td>R@1</td>
        <td>R@2</td>
        <td>R@5</td>
    </tr>
    <tr>
        <td>DGMN (Original)</td>
        <td>65.6</td>
        <td>78.3</td>
        <td>91.2</td>
        <td>67.6</td>
        <td>80.2</td>
        <td>92.9</td>
        <td>58.8</td>
        <td>62.5</td>
        <td>87.7</td>
    </tr>
    <tr>
        <td>DGMN (Reproduced)</td>
        <td>71.2</td>
        <td>85.1</td>
        <td>96.5</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>DGMN + 100d_w2v (Reproduced)</td>
        <td>72.5</td>
        <td>85.9</td>
        <td>97.1</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>

## Datasets
You can download the datasets and their corresponding embedding tables used in their paper from the following links.
+ [PERSONA-CHAT](https://drive.google.com/open?id=1gNyVL5pSMO6DnTIlA9ORNIrd2zm8f3QH) and its [embedding and vocabulary files](https://drive.google.com/open?id=1gGZfQ-m7EGo5Z1Ts93Ta8GPJpdIQqckC). <br>
+ [CMU_DoG](https://drive.google.com/file/d/1GYKelOS9_yvc66fe9NqMnxWAwYAfoIzP/view?usp=sharing) and its [embedding and vocabulary files](https://drive.google.com/file/d/1vCm2shBE2ZxPI1Vw6bmCVv3xujVL72Xs/view?usp=sharing). <br>

Unzip the datasets to the folder of `dataset` and run the preprocessing codes provided in [JasonForjoy/FIRE](https://github.com/JasonForJoy/FIRE).

Then, you will obtain the preprocessing files in `dataset/personachat_preprocessed` and `dataset/cmudog_preprocessed`.

## Train
Temporarily we only test on CMUDoG. Here is an example to train DGMN in CMUDoG dataset.
```shell
sh run_cmudog.sh
```

## Acknowledgements
Thank Xueliang Zhao for providing the source codes written in TensorFlow for reference, it does help a lots.

## TODO
+ Evaluate on PersonaChat dataset.
+ Check why the reproduced version outperforms the reported result reported in the original paper.
