#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

class TestCase {
public:
	TestCase() = default;
	~TestCase() = default;

	bool Init(std::string model_path);
	cv::Size size GetInputSize();
	void RunModel();
};
