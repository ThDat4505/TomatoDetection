#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Constants configuration for YOLOv8 (Laboro Tomato dataset classes or general coco)
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.40f;

// Example class names for tomato maturity (adjust based on your training labels)
std::vector<std::string> class_names = {
    "fully_ripened", "half_ripened", "green"
};

cv::Mat format_to_square(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

int main(int argc, char **argv) {
    // 1. Load the exported YOLOv8 ONNX model
    std::string model_path = "best.onnx"; // Path to your exported ONNX model
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Set backend to OpenCV (use CUDA if available and compiled with it)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Change to DNN_TARGET_CUDA if using GPU

    // 2. Open the Web Camera (0 is usually the default integrated webcam)
    cv::VideoCapture capture(0, cv::CAP_DSHOW);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::cout << "Starting camera loop. Press 'Esc' to exit." << std::endl;

    while (capture.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Blank frame grabbed. Exiting..." << std::endl;
            break;
        }

        // 3. Preprocess frame for YOLOv8
        cv::Mat square_image = format_to_square(frame);
        cv::Mat blob;
        cv::dnn::blobFromImage(square_image, blob, 1.0 / 255.0, 
                               cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 
                               cv::Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        cv::Mat outputs = net.forward();

        // YOLOv8 output shape handling: [1, 84, 8400] (for 80 classes + 4 box coords, or custom)
        // Reshape or parse outputs depending on dimensions
        int dimensions = outputs.size[1]; // e.g., 4 + classes
        int rows = outputs.size[2];       // e.g., 8400 boxes
        
        // If dimensions is 3D, flatten/reposition for processing
        outputs = (outputs.size[1] > 1 ? outputs.reshape(1, dimensions) : outputs);
        // Note: Depending on OpenCV version, standard YOLOv8 output parsing layout requires transpose:
        // Transpose output from [1, 84, 8400] to [1, 8400, 84]
        cv::Mat det_output(outputs.size[1], outputs.size[2], CV_32F, outputs.ptr<float>());
        // Simplified parser loop variables
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = (float)square_image.cols / INPUT_WIDTH;
        float y_factor = (float)square_image.rows / INPUT_HEIGHT;

        // Note: YOLOv8 raw format parser 
        // For standard models, data format is pointer-driven:
        // Pointer row layout: [x_center, y_center, width, height, class_scores...]
        // Below is a general structural traversal:
        int num_channels = outputs.size[1];
        int num_anchors = outputs.size[2];
        
        // Correct structure conversion for OpenCV DNN block:
        cv::Mat p_data = outputs.reshape(1, num_channels); 
        // A safer robust traversal loop for standard object detection output:
        // (Assuming standard row layout [8400, attributes])
        // Let's transpose data for easier row-by-row iteration if required:
        cv::Mat transposed_output;
        cv::transpose(outputs.reshape(1, num_channels), transposed_output);

        // Iterate through detections
        for (int i = 0; i < transposed_output.rows; ++i) {
            float* data = transposed_output.ptr<float>(i);
            float confidence = data[4]; // Adjust index based on your custom class layout start

            // If you have multiple classes, find the max class score index starting at index 4
            // For single class or multi-class:
            float max_class_score = 0;
            int class_id = 0;
            
            // Assuming 3 classes (ripe, half-ripe, green) -> Total channels = 4 (box) + 3 (classes) = 7
            for (int c = 0; c < transposed_output.cols - 4; ++c) {
                float class_score = data[4 + c];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    class_id = c;
                }
            }

            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(max_class_score);
                class_ids.push_back(class_id);

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((cx - 0.5 * w) * x_factor);
                int top = static_cast<int>((cy - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        // Perform Non-Maximum Suppression to filter overlapping duplicate boxes
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            
            // Draw bounding box
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            // Draw label text
            std::string label = (class_ids[idx] < class_names.size()) ? 
                                class_names[class_ids[idx]] : "tomato";
            label += ":" + cv::format("%.2f", confidences[idx]);
            
            cv::putText(frame, label, cv::Point(box.x, box.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }

        // Display live camera feed with detections
        cv::imshow("Tomato Detection - YOLOv8 C++", frame);

        // Exit loop if 'Esc' key is pressed
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}