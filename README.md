# Isaac ROS Image Segmentation

NVIDIA-accelerated, deep learned semantic image segmentation

<div align="center"><img alt="sample input to image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example.png/" width="320px"/>
<img alt="sample output from image segmentation" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_seg.png/" width="320px"/></div>

## Overview

Isaac ROS Image Segmentation contains ROS packages for semantic image segmentation.

These packages provide methods for classification of an input image
at the pixel level by running GPU-accelerated inference on a DNN model.
Each pixel of the input image is predicted to belong to a set of defined classes.
The output prediction can be used by perception functions to understand where each
class is spatially in a 2D image or fuse with a corresponding depth location in a 3D scene.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_nodegraph.png/" width="500px"/></a></div>

| Package                                                                                                                                                                  | Model Architecture                                | Description                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|--------------------------------------------------------------------------|
| [Isaac ROS U-NET](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)                        | [U-NET](https://en.wikipedia.org/wiki/U-Net)      | Convolutional network popular for biomedical imaging segmentation models |
| [Isaac ROS Segformer](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#quickstart)               | [Segformer](https://arxiv.org/abs/2105.15203)     | Transformer-based network that works well for objects of varying scale   |
| [Isaac ROS Segment Anything](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#quickstart) | [Segment Anything](https://segment-anything.com/) | Segments any object in an image when given a prompt as to which one      |

Input images may need to be cropped and resized to maintain the aspect ratio and match the input
resolution expected by the DNN model; image resolution may be reduced to improve
DNN inference performance, which typically scales directly with the
number of pixels in the image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_image_segmentation_example_bboxseg.png/" width="320px"/></a></div>

Image segmentation provides more information and uses more compute than
object detection to produce classifications per pixel, whereas object
detection classifies a simpler bounding box rectangle in image
coordinates. Object detection is used to know if, and where spatially in
a 2D image, the object exists. On the other hand, image segmentation is used to know which
pixels belong to the class. One application is using the segmentation result, and fusing it with the corresponding depth
information in order to know an object location in a 3D scene.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                                                  | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                                   | Orin NX<br/><br/>                                                                                                                                                   | Orin Nano Super 8GB<br/><br/>                                                                                                                                         | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                        |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_segment_anything_graph.py)<br/><br/><br/>Full SAM<br/><br/>          | 720p<br/><br/><br/><br/> | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_sam_graph-agx_orin.json)<br/><br/><br/>390 ms @ 30Hz<br/><br/>        | –<br/><br/><br/><br/>                                                                                                                                               | –<br/><br/><br/><br/>                                                                                                                                                 | [14.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_sam_graph-x86-4090.json)<br/><br/><br/>74 ms @ 30Hz<br/><br/>        |
| [SAM Image Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_segment_anything_benchmark/scripts/isaac_ros_mobile_segment_anything_graph.py)<br/><br/><br/>Mobile SAM<br/><br/> | 720p<br/><br/><br/><br/> | [8.40 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-agx_orin.json)<br/><br/><br/>120 ms @ 30Hz<br/><br/> | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-orin_nx.json)<br/><br/><br/>240 ms @ 30Hz<br/><br/> | [2.22 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-orin_nano.json)<br/><br/><br/>230 ms @ 30Hz<br/><br/> | [62.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_mobile_sam_graph-x86-4090.json)<br/><br/><br/>22 ms @ 30Hz<br/><br/> |
| [TensorRT Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_unet_benchmark/scripts/isaac_ros_unet_graph.py)<br/><br/><br/>PeopleSemSegNet<br/><br/>                                         | 544p<br/><br/><br/><br/> | [436 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-agx_orin.json)<br/><br/><br/>10 ms @ 30Hz<br/><br/>         | [212 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nx.json)<br/><br/><br/>13 ms @ 30Hz<br/><br/>         | [224 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-orin_nano.json)<br/><br/><br/>13 ms @ 30Hz<br/><br/>         | [587 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_unet_graph-x86-4090.json)<br/><br/><br/>3.7 ms @ 30Hz<br/><br/>       |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_segformer`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segformer/index.html#api)
* [`isaac_ros_segment_anything`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_segment_anything/index.html#api)
* [`isaac_ros_unet`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_segmentation/isaac_ros_unet/index.html#api)
 
  People Avoidance and Following in Isaac Sim Using Isaac ROS U-NET Segmentation
Overview
Isaac ROS Image Segmentation includes ROS packages designed for semantic image segmentation. These packages enable pixel-level classification of input images by performing GPU-accelerated inference with deep neural network models. Each pixel in the input image is classified into predefined categories, allowing perception systems to determine the spatial distribution of each class within a 2D image. This output can also be integrated with depth information to create a 3D scene understanding.
DNN Model Overview of PeopleSemSegNet for Autonomous Mobile Robots (AMRs)
The PeopleSemSegNet model is designed for autonomous mobile robots (AMRs) to detect and segment human figures within images. This model provides a high level of accuracy in identifying one or more people in various environments, making it highly suitable for commercial applications.
Key Features
Human Detection and Segmentation:

PeopleSemSegNet is trained to detect and segment "person" objects in images. It generates a semantic segmentation mask that outlines all detected individuals with high precision, allowing the AMR to differentiate people from other objects in the scene.

2. Semantic Segmentation Mask:
The model outputs a detailed segmentation mask where each pixel is classified as either belonging to a person or the background. This mask is essential for tasks that require precise localization and interaction with people.

3. Applications in AMRs:
PeopleSemSegNet enhances the capabilities of AMRs by enabling functionalities such as:
People Avoidance: Allowing the robot to navigate around detected people to avoid collisions.
People Following: Directing the robot to follow a specific person or interact with them based on the segmentation mask.
Human-Robot Interaction: Facilitating natural and safe interactions with humans in various environments, such as offices, hospitals, or public spaces.

4. Integration with ROS:
In an Isaac ROS environment, the model's output is typically published on a topic such as /unet/colored_segmentation_mask. This integration allows other ROS nodes to subscribe to this topic, process the segmentation data, and make decisions based on the detected human figures.

Why People Avoidance and Following are Important:
Discuss use cases like warehouse robots, autonomous delivery systems, and social robots.
Warehouse Robots:

People Avoidance: In a warehouse setting, robots often operate in dynamic environments with human workers moving around. People avoidance is crucial to prevent collisions, ensuring the safety of both the robots and the workers. Robots equipped with people avoidance capabilities can navigate around obstacles and adjust their paths to avoid interfering with human activities.
People Following: Some warehouse robots may be designed to follow human operators, such as inventory pickers or supervisors. By following these individuals, robots can assist with carrying tools, restocking items, or even providing real-time inventory updates, enhancing productivity and streamlining operations.

2. Autonomous Delivery Systems:
People Avoidance: Delivery robots operating in public spaces or residential areas must be able to navigate around pedestrians to avoid accidents and ensure safe delivery. People avoidance technology allows these robots to detect and avoid individuals, minimizing the risk of collisions and improving overall service quality.
People Following: In some scenarios, autonomous delivery robots may need to follow a specific person, such as a delivery recipient, to complete a transaction or hand over a package. This capability is particularly useful in environments where the recipient is moving or when the delivery process requires personal interaction.

figure 1In Figure 1, the Isaac Sim simulation demonstrates the integration of Isaac ROS Unet image segmentation for people following. When the topic /unet/colored_segmentation_mask is activated, it provides segmentation information where people are highlighted in red. Using this segmentation data, the robot is able to identify and track individuals within the warehouse environment, allowing it to follow humans effectively.
Key Isaac ROS Packages for People Avoidance and Following:
1. isaac_ros_image_pipeline:
This package provides a suite of tools for image processing and segmentation. It enables the robot to visually recognize and interpret its surroundings, which is crucial for tasks such as people avoidance and following. By processing camera feeds and segmenting the images, this package helps the robot identify and track objects or individuals within its environment.

2. isaac_ros_unet:
The isaac_ros_unet package leverages U-Net-based segmentation models to perform semantic segmentation. U-Net is a deep learning architecture specifically designed for image segmentation tasks, allowing the robot to distinguish and segment different objects in an image. When used for people detection, the U-Net model can accurately identify and outline individuals by assigning a specific class label (e.g., people) to each pixel. This capability enables the robot to follow or avoid people based on the segmented output, where individuals are highlighted distinctly (e.g., in red).

4. isaac_ros_dnn_image_encoder:
This package is designed for encoding input images to facilitate efficient processing by deep neural network models. The isaac_ros_dnn_image_encoder prepares and transforms camera feeds into a format suitable for deep learning algorithms, such as those used in isaac_ros_unet. This preprocessing step ensures that the segmentation models receive high-quality and appropriately formatted input, leading to more accurate and reliable detection and segmentation of people or other objects.
figure 2In Figure 2 & 3, the RQT graph illustrates the interaction between key Isaac ROS packages involved in people avoidance and following:
isaac_ros_image_pipeline: This package is responsible for processing and segmenting image data from the robot's sensors. It prepares the image feed for further analysis and segmentation.
isaac_ros_unet: The U-Net-based segmentation model is used to perform semantic segmentation on the processed images. This model identifies and classifies objects, such as people, within the images based on the segmentation mask provided.
isaac_ros_dnn_image_encoder: This package encodes the input images to make them compatible with deep neural network models. It ensures that the images are formatted correctly for accurate and efficient processing by the isaac_ros_unet model.

figure 3figure 4In Figure 4 , the Isaac Sim simulation demonstrates the integration of Isaac ROS Unet image segmentation for people avoidance. When the topic /unet/colored_segmentation_mask is activated, it provides segmentation information with people highlighted in red. Using this segmentation data, the robot can detect the presence of individuals within the warehouse environment and adjust its path to avoid collisions, ensuring safe and efficient navigation around humans.
Advantages of Using the isaac_ros_unet Image Segmentation Package with the PeopleSemSegNet AMR Model.
Accurate People Detection:

The isaac_ros_unet package uses the U-Net model, which is highly accurate for identifying people in images. This ensures the robot can reliably detect individuals.
2. Real-Time Processing:
The package allows the robot to process and segment images in real time. This is crucial for quickly reacting to changes in a busy environment like a warehouse.
3. Enhanced Safety:
By accurately detecting people, the robot can avoid collisions and navigate safely around individuals, improving safety for both the robot and people.
4. Better Efficiency:
With precise people detection, the robot can plan its movements more effectively, leading to faster and more efficient operations in environments like warehouses.
Conclusion
The integration of the `isaac_ros_unet` image segmentation package with the PeopleSemSegNet model significantly enhances the capabilities of autonomous mobile robots (AMRs) in dynamic environments. By leveraging high-accuracy semantic segmentation, real-time processing, and seamless ROS integration, robots can effectively detect and avoid people, ensuring safe and efficient navigation. The flexibility and customization options of the `isaac_ros_unet` package make it a valuable tool for adapting to various operational needs and environments. Overall, this technology advances the safety and operational efficiency of AMRs, paving the way for smarter and more reliable robotics solutions in settings such as warehouses and other complex environments.

## Latest

Update 2024-12-10: Update to be compatible with JetPack 6.1
