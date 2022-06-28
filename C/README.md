# Mirco Speech Example 
This example is based on the Tensorflow micro-speech example, which demonstrates a basic keyword recognition application that can detect "yes" & "no". The purpose of this simple example is to showcase the usage of the AI Infrastructure elements in place to build an end-to-end Machine Learning example. 

A full documentation of the example is hosted on [Confluence](https://confluencewikiprod.intra.infineon.com/display/AIML/Micro-speech+example).
## Testing

The whole timing of the application is controlled with the latest_audio_timestamp_ms. This parameter is updated by the ISR of the audio provider. In the main_function the current timestamp is needed for calculating the input features of the network. The current timestamp is defined by get_latest_audio_timestamp. So when controlling the latest_audio_timestamp_ms you also can control the whole timing. This is the base for end to end testing. 

The test_audio_provider project verifies that the audio provider works correct. So we do not test the audio provider again and mok it. Due to the timing of semihosting we need to control the latest_audio_timestamp_ms. 

To run the application in test mode you need to run the build with ```TEST_MODE=1```:

```bash
make build -j TEST_MODE=1
```

