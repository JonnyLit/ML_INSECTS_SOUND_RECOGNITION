#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread> // For simulating a function with sleep()
#include <complex>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnn/ArmNN.hpp>
#include <filesystem>

#include <cstdlib> // for atoi
#include <string>
#include <map>
#include <algorithm>
#include <cmath> // for exp
#include <cstring>



namespace fs = std::filesystem;





struct FileInfo {
    fs::path path;
    int segmentNumber;

    FileInfo(const fs::path& p, int seg) : path(p), segmentNumber(seg) {}
};






// Function to extract main filename and segment number
bool parseFilename(const std::string& filename, std::string& mainPart, int& segmentNumber) {
    // Example:
    
    // From filename: ESP32_model_1_(Hive1)_08-10-2025_11-09-19_segment_0.bin
    // to filename: ESP32_model_1_(Hive1)_08-10-2025_11-09-19_queen_absent_segment_0.bin

    // Find the position of "_segment_" in the filename
    size_t segmentPos = filename.find("_segment_");
    if (segmentPos == std::string::npos) {
        return false; // Not matching pattern
    }

    // Extract main part (before "segment_")
    mainPart = filename.substr(0, segmentPos);

    // Extract the segment number (after "_segment_")
    size_t segmentNumberStart = segmentPos + strlen("_segment_");
    size_t segmentNumberEnd = filename.find('.', segmentNumberStart);
    if (segmentNumberEnd == std::string::npos) {
        return false; // No extension found
    }

    std::string segmentStr = filename.substr(segmentNumberStart, segmentNumberEnd - segmentNumberStart);
    try {
        segmentNumber = std::stoi(segmentStr);
    } catch (...) {
        return false; // Conversion failed
    }

    return true;
}





// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData){
	return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
    }

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData){
	return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
    }


#include <cmath> // for exp
#include <vector> // assuming outputData is std::vector<float>


std::vector<float> softmax(const std::vector<float>& input, std::vector<float>& output) {
    // Resize output to match input size
    output.resize(input.size());

    // Find max value for numerical stability
    float maxVal = *std::max_element(input.begin(), input.end());

    float sumExp = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal); // for numerical stability
        sumExp += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sumExp;
    }

    // Print the softmax probabilities
    std::cout << "Softmax probabilities: ";
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return output;
}










// Function to run inference using an ARMNN model
int RunInference(const std::string& modelPath, std::vector<float> inputData, const std::string& backendType, const unsigned int rows, const unsigned int cols, std::vector<float>& list_of_softmax_counters_per_class_results, armnn::NetworkId networkId, armnn::Status status, armnn::IRuntimePtr& runtime, armnnTfLiteParser::BindingPointInfo inputBindingInfo, armnnTfLiteParser::BindingPointInfo outputBindingInfo) {
    std::cout << "RunInference>" << std::endl;
    

    // Ensure input data is correct
    std::cout << "    Ensure input data is correct" << std::endl;
    if (inputData.size() != cols * rows) { // Check if the size is correct
        std::cerr << "Input data size is incorrect. Expected: " << (cols * rows) << ", Got: " << inputData.size() << std::endl;
        return -1;
    }
    
    // Create TensorInfo for input, ensuring it's a constant tensor
    std::cout << "    Create TensorInfo for input, ensuring it's a constant tensor" << std::endl;
    armnn::TensorInfo inputTensorInfo({1, rows, cols, 1}, armnn::DataType::Float32);

    // Create ConstTensor with the correct TensorInfo
    std::cout << "    Create ConstTensor with the correct TensorInfo" << std::endl;
    
    // Create InputTensors
    std::cout << "    Create InputTensors" << std::endl;
    armnn::InputTensors inputTensor = MakeInputTensors(inputBindingInfo, inputData.data());
    try {
        inputTensorInfo.SetConstant(true);
    } catch (const armnn::InvalidArgumentException& e) {
        std::cerr << "Failed to create ConstTensor: " << e.what() << std::endl;
      return -1;
   	}



    // Allocate memory for output tensor data (4 classes)
    std::cout << "    Allocate memory for output tensor data (4 classes)" << std::endl;
    std::vector<float> outputData(4);


    // Create OutputTensors
    std::cout << "    Create OutputTensors" << std::endl;
    armnn::OutputTensors outputTensor = MakeOutputTensors(outputBindingInfo, outputData.data());





    
    std::cout << "Perform inference (runtime->EnqueueWorkload(networkId, inputTensor, outputTensor);)" << std::endl;
    auto start_inference = std::chrono::high_resolution_clock::now();
    // Run the inference
    
    // Perform inference
    status = runtime->EnqueueWorkload(networkId, inputTensor, outputTensor);
    if (status != armnn::Status::Success) {
        std::cerr << "Inference failed!" << std::endl;
        return -1;
    }
    
    // Record the inference end time
    auto end_inference = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
    std::cout << "Elapsed inference time: " << duration_inference.count() << " milliseconds" << std::endl;
    
    
    
    
    std::vector<float> rawOutput(4);
    std::vector<float> probabilities(4);
    for (int i = 0; i < 4; ++i) {
		rawOutput[i] = outputData[i]; // raw model outputs
    }

    std::vector<float> temp_list_of_softmax_counters_per_class_results = softmax(rawOutput, probabilities);
    // Now 'probabilities' sum to 1 and can be used for classification
	
    for (int i = 0; i < 4; ++i) {
		list_of_softmax_counters_per_class_results[i] += temp_list_of_softmax_counters_per_class_results[i];
    }
	

    // Assuming outputData contains the results from the model
    std::cout << "Output Data: ";
    for (float val : outputData) {
        std::cout << val << " ";  // Print raw output values
    }
    std::cout << std::endl;
    
    
    /*
    std::cout<< "inputBindingInfo.second.GetQuantizationScale(): " << inputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< "inputBindingInfo.second.GetQuantizationOffset(): " << inputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< "inputBindingInfo.second.GetNumBytes(): " << inputBindingInfo.second.GetNumBytes() << std::endl;

    std::cout<< "outputBindingInfo.second.GetQuantizationScale(): " << outputBindingInfo.second.GetQuantizationScale() << std::endl;
    std::cout<< "outputBindingInfo.second.GetQuantizationOffset(): " << outputBindingInfo.second.GetQuantizationOffset() << std::endl;
    std::cout<< "outputBindingInfo.second.GetNumBytes(): " << outputBindingInfo.second.GetNumBytes() << std::endl;
    */

    // Process the output
    std::cout << "    Process the output" << std::endl;
    //int predicted_class = std::distance(outputData.begin(), std::max_element(outputData.begin(), outputData.end()));
    int predicted_class = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
    std::cout << "Predicted class: " << predicted_class << std::endl; // Will output 0 (queen_absent), 1 (queen_present_newly_accepted), 2 (queen_present_original), 3 (queen_present_rejected)

    return predicted_class; // return the classification result
}






// Function to insert a custom substring after mainPart in filename
bool insertSubstringAfterMainPart(const fs::path& filepath, const std::string& mainPart, const std::string& insertStr) {
    std::string filename = filepath.filename().string();
    std::string extension = filepath.extension().string();

    // Remove extension
    std::string nameWithoutExt = filename.substr(0, filename.size() - extension.size());

    // Find position of mainPart
    size_t mainPos = nameWithoutExt.find(mainPart);
    if (mainPos == std::string::npos) {
        std::cerr << "Main part not found in filename: " << filename << std::endl;
        return false;
    }

    // Position where mainPart ends
    size_t insertPos = mainPos + mainPart.length();

    // Insert the custom substring after mainPart
    std::string newName = nameWithoutExt.substr(0, insertPos) + insertStr + nameWithoutExt.substr(insertPos);

    // Add extension back
    newName += extension;

    fs::path newPath = filepath.parent_path() / newName;

    try {
        fs::rename(filepath, newPath);
        std::cout << "Renamed: " << filepath << " -> " << newPath << std::endl;
        return true;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error renaming: " << e.what() << std::endl;
        return false;
    }
}






int main(int argc, char** argv) {
    
    // Record the overall start time
    auto start_overall = std::chrono::high_resolution_clock::now();
    
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <model.tflite> <spectrograms directoryPath> <rename_files 0(no) 1(yes)> <backend type (CpuRef or CpuAcc)>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    const char* directoryPath = argv[2];
    int rename_files_0_1 = std::atoi(argv[3]);
    std::string backendType = argv[4];
    
    std::string dirPathStr(directoryPath); // Convert to std::string
    
    
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "executable: " << argv[0] << std::endl;
    std::cout << "Input arguments: " << std::endl;
    std::cout << "modelPath: "<< modelPath << ", spectrograms directoryPath: " << dirPathStr << ", rename_files_0_1: " << rename_files_0_1 << ", backendType: " << backendType <<std::endl;


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // For .npy files
    // Assume you have dimensions to read back
    const int rows = 32;
    const int cols = 44;
    int num_classes = 4; // 4 output classes for classification

    // Define the directory containing the .bin files
    //const std::string directoryPath = "path/to/your/directory";



    std::vector<float> list_of_softmax_counters_per_class_results(num_classes); // creates a list of num_class floats initialized to 0.0


	
    // Create and open the file for writing
    std::string txt_file_path = dirPathStr + "/Hives_Classes.txt";
    std::ofstream outFile(txt_file_path);
    if (!outFile) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return 1;
    }



    // Map from main filename to its segments
    std::map<std::string, std::vector<FileInfo>> filesByMain;

    for (const auto& entry : fs::directory_iterator(dirPathStr)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".wav" || entry.path().extension() == ".bin")) {
            std::string filename = entry.path().filename().string();
            std::string mainPart;
            int segmentNumber;
            if (parseFilename(filename, mainPart, segmentNumber)) {
                filesByMain[mainPart].push_back(FileInfo(entry.path(), segmentNumber));
            }
        }
    }

    // For each main file group, sort segments by segment number
    for (auto& [mainPart, fileList] : filesByMain) {
        std::sort(fileList.begin(), fileList.end(), [](const FileInfo& a, const FileInfo& b) {
        	return a.segmentNumber < b.segmentNumber;
        });
    }












    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //LOADING THE MODEL NETWORK
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);

    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();

    // Read the TFLite model file
    std::cout << "    Read the TFLite model file" << std::endl;
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open model file: " << modelPath << std::endl;
        return -1;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> modelData(size);
    file.read(reinterpret_cast<char*>(modelData.data()), size);
    file.close();

    // Create the network from the binary data
    std::cout << "    Create the network from the binary data" << std::endl;
    armnn::INetworkPtr network = parser->CreateNetworkFromBinary(modelData);
       
    // Return the input tensor names for a given subgraph
    std::vector<std::string> InputBindingName = parser->GetSubgraphInputTensorNames(0);

    // Return the output tensor names for a given subgraph
    std::vector<std::string> OutputBindingName = parser->GetSubgraphOutputTensorNames(0);
    
    std::cout << "    InputBindingName[0] = " << InputBindingName[0] << std::endl;
    std::cout << "    OutputBindingName[0] = " << OutputBindingName[0] << std::endl;
    
    
    size_t numSubgraph = parser->GetSubgraphCount();
    std::cout << "    numSubgraph = " << numSubgraph << std::endl;
    
    for(size_t iter_subgraphs = 0; iter_subgraphs < numSubgraph; iter_subgraphs++){
        std::cout << "    iter_subgraphs = " << iter_subgraphs << std::endl;
        std::cout << "    InputBindingName = " << InputBindingName[iter_subgraphs] << std::endl;
        std::cout << "    OutputBindingName = " << OutputBindingName[iter_subgraphs] << std::endl;
    }
    
    // Find the binding points for the input and output nodes
    armnnTfLiteParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo(0, InputBindingName[0]);
    armnnTfLiteParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, OutputBindingName[0]);










    // Optimize the network
    std::cout << "    Optimize the network" << std::endl;
    std::vector<armnn::BackendId> backends = {backendType};  // Change based on available backends
    armnn::OptimizerOptionsOpaque optimizerOptionsOpaque;  // Use the ABI stable variant

    armnn::IOptimizedNetworkPtr optimizedNetwork = armnn::Optimize(*network, backends, runtime->GetDeviceSpec(), optimizerOptionsOpaque);
    if (!optimizedNetwork)
    {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        return -1;
    }

    // Load the optimized network
    std::cout << "    Load the optimized network" << std::endl;
    armnn::NetworkId networkId;
    armnn::Status status = runtime->LoadNetwork(networkId, std::move(optimizedNetwork));
    if (status != armnn::Status::Success) {
        std::cerr << "Failed to load network!" << std::endl;
        return -1;
    }

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



















    // Process files in order: for each main file, process segments in order
    for (const auto& [mainPart, fileList] : filesByMain) {
    	std::cout << "mainPart: " << mainPart << std::endl;
        
		// Record the time for a main part
		auto start_mainPart = std::chrono::high_resolution_clock::now();
    
 
		// Find the position of the first '('
		size_t startPos = mainPart.find('(');
		// Find the position of the first ')' after startPos
		size_t endPos = mainPart.find(')', startPos);
				
		std::string Hive = "";
		if (startPos != std::string::npos && endPos != std::string::npos && endPos > startPos) {
			// Extract the substring (Hive) between '(' and ')'
			Hive = mainPart.substr(startPos + 1, endPos - startPos - 1);
			std::cout << "Hive: " << Hive << std::endl;
		} else {
			std::cout << "Brackets not found or improperly placed." << std::endl;
		}
		    
        
        
        
    	// Get the number of files for this mainPart
    	size_t fileCount = fileList.size();
    	std::cout << "Number of files for this mainPart: " << fileCount << std::endl;




    	for (int iter_current_segments = 0; iter_current_segments < fileCount; iter_current_segments++) {
	    	std::cout << "segment: " << fileList[iter_current_segments].path << std::endl;
    	}



    	list_of_softmax_counters_per_class_results = std::vector<float>(num_classes, 0.0f); //initialized to zero

		int class_index;
        for (const auto& fileInfo : fileList) {
            // Access your file here
            if (fileInfo.path.extension() == ".bin") {
				std::cout << "\n\n----------------------------------------------" << std::endl;
				std::cout << "Processing: " << fileInfo.path << std::endl;


				// Get the path to your binary file
				const char* filename = fileInfo.path.c_str();
						
				// Open the binary file in input mode
				std::ifstream file(filename, std::ios::binary);
				if (!file) {
					std::cerr << "Error: Could not open the file!" << std::endl;
					return 1;
				}

				// Use a vector to store the loaded data
				std::vector<float> image;

				// Read the data. Assuming you know the number of elements.
				// You might want to read this from metadata in a real application.
				size_t num_elements = rows*cols; // Change this according to your needs
				image.resize(num_elements);

				// Read the binary data into the vector
				file.read(reinterpret_cast<char*>(image.data()), num_elements * sizeof(float));
						
				// Check if the reading was successful
				if (file.gcount() != num_elements * sizeof(float)) {
					std::cerr << "Error: Could not read enough data!" << std::endl;
					return 1;
				}

				// Close the file
				file.close();
						


				// Output the matrix dimensions and data for verification
				std::cout << "Matrix size: " << image.size() << std::endl; // Should output the product of 44 x 32 --> 1408


				// Here, you can use input_tensor with your ArmNN model for inference
				int predicted_class;
				// Record the inference start time
				auto start_inference = std::chrono::high_resolution_clock::now();
				// Run the inference
				//RunInference(modelPath, input_data, backendType);
						
				predicted_class = RunInference(modelPath, image, backendType, (unsigned int)rows, (unsigned int)cols, list_of_softmax_counters_per_class_results, networkId, status, runtime, inputBindingInfo, outputBindingInfo);
						
				// Record the inference end time
				auto end_inference = std::chrono::high_resolution_clock::now();

						
				// Calculate the duration in milliseconds
				auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
				std::cout << "Elapsed inference time (RunInference): " << duration_inference.count() << " milliseconds" << std::endl;

			}

		}
		    
		    
		    
	
		std::cout << "\nCumulative softmax classification values:" << std::endl;
		float max = 0.0;
			
		for (int iter_list=0; iter_list<num_classes; iter_list++) {
			std::cout << "Class " << iter_list << " value: " << list_of_softmax_counters_per_class_results[iter_list] << std::endl;
			if (list_of_softmax_counters_per_class_results[iter_list] >= max){
				max = list_of_softmax_counters_per_class_results[iter_list];
				class_index = iter_list;
			}
		}
			
		std::cout << "" << std::endl;
		    
		    
		std::string substringToInsert = ""; // label to insert in the filename
		std::string result_class = ""; // to insert as value in a key:value contest inside the Hives_Classes.txt file
		if (class_index == 0){
			substringToInsert = "_<queen_absent>";
			result_class = "queen_absent";
		}else if (class_index == 1){
			substringToInsert = "_<queen_present_newly_accepted>";
			result_class = "queen_present_newly_accepted";
		}else if (class_index == 2){
			substringToInsert = "_<queen_present_original>";
			result_class = "queen_present_original";
		}else if (class_index == 3){
			substringToInsert = "_<queen_present_rejected>";
			result_class = "queen_present_rejected";
		}
		std::cout << "Final predicted class: " << substringToInsert << "(" << class_index << ")\n\n" << std::endl;


		if (rename_files_0_1 == 1) {
			for (int iter_group_of_files = 0; iter_group_of_files < fileCount; iter_group_of_files++) {
				// std::cout << "segment: " << fileList[iter_group_of_files].path << std::endl; // commented for non verbose mode ****************************************************************************************
				// Call the function with variable insert substring
				insertSubstringAfterMainPart(fileList[iter_group_of_files].path, mainPart, substringToInsert);
			}
		}
		
		outFile << Hive << ":" << result_class << std::endl;

		
		// Record the time for a main part
		auto end_mainPart = std::chrono::high_resolution_clock::now();
		
		// Calculate the duration in milliseconds
		auto duration_mainPart = std::chrono::duration_cast<std::chrono::milliseconds>(end_mainPart - start_mainPart);
		std::cout << "Elapsed mainPart time: " << duration_mainPart.count() << " milliseconds" << std::endl;
		std::cout << "////////////////////////////////////////////////////////////\n\n" << std::endl;
    }
    
    
    // Record the overall end time
    auto end_overall = std::chrono::high_resolution_clock::now();
    
    // Calculate the duration in milliseconds
    auto duration_overall = std::chrono::duration_cast<std::chrono::milliseconds>(end_overall - start_overall);
    std::cout << "Elapsed overall time: " << duration_overall.count() << " milliseconds" << std::endl;
		
} 
      


