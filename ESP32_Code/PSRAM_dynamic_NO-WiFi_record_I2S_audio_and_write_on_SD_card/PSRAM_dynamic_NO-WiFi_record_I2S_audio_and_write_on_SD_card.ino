#include "Arduino.h"
#include <WiFi.h>
#include <AsyncMqttClient.h>
#include "driver/i2s.h"
#include <ArduinoJson.h> // Include this for JSON parsing
#include "device.h"
#include <time.h>
#include <esp_system.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_sleep.h"
#include "sd_read_write.h"
#include "SD_MMC.h"

#define SD_MMC_CMD 15 //Please do not modify it
#define SD_MMC_CLK 14 //Please do not modify it
#define SD_MMC_D0  2  //Please do not modify it

// NTP Server
const char* ntpServer = "pool.ntp.org";
// Time zone offset in seconds (UTC+0), adjust as needed
const long gmtOffset_sec = 3600; // Change this based on your timezone
const int daylightOffset_sec = 3600; // Change this if you observe Daylight Saving

#define DEBUG_MODE


//Device object
Device esp32_device; // class defined in the "Device.h" header
// Global or static variable to communicate between tasks
TaskHandle_t i2sTaskHandle;
bool isTaskRunning = false;


String location = "Hive1";



//////////////////////////I2S PARAMETERS//////////////////////////////

#define CONFIG_SPIRAM_ENABLE

//le porte GPIO corrette per l'ESP32 WROOM 32D e l'ESP32 WROVER-E sono: I2S_WS 15, I2S_SD 13, I2S_SCK 2
#define I2S_WS 15 //15, 25
#define I2S_SD 13 //13, 33
#define I2S_SCK 2 //2, 32
#define I2S_PORT I2S_NUM_0
#define L_R_channel_selector 12

//The following variables have been defined as class members in device.h
//#define I2S_SAMPLE_RATE   (22050)
//#define I2S_SAMPLE_BITS   (16)
//#define RECORD_TIME       (1) //Seconds
//#define I2S_CHANNEL_NUM   (1)
//#define AUDIO_RECORD_SIZE (I2S_CHANNEL_NUM * I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 * RECORD_TIME) //5 sec 22.05KHz --> 220'500
//#define I2S_READ_LEN      (AUDIO_RECORD_SIZE / 20) // to work properly, I2S_READ_LEN must be chosen in such a way that AUDIO_RECORD_SIZE / I2S_READ_LEN equals an integer value


const int headerSize = 44;
int audio_data_size = headerSize + esp32_device.get_audio_record_size();
char* audio_data;
String prefix_audio_name;  //"prefix_audio_name" is a 5-characters random generated alphanumeric string, so different every time we turn on-off the esp32.
                          //The audio will be named and saved as the union between the values of "prefix_audio_name", plus a separator "_", 
                          //and plus a number "record_counter" that increases with the recordings, so of value between "1 and +inf" (till the SD-Card memory allows). 
                          //For example, if I turn on the esp32, and the string randomly generated is "ab123", its audioS will be named like: ab123_1.wav, ab123_2.wav, ab123_3.wav, ab123_4.wav, ab123_5.wav, ...
                          //In this way, knowing a priori the time we turned on the esp32, and knowing how many seconds elapse between consecutive recordings, 
                          //we can in retrospect reinterpret the audio records names, thus avoiding to use WiFi.
int record_counter = 0;   


void reportMemoryStats() {
    //size_t freeHeap = esp_get_free_heap_size();
    size_t freeHeap = ESP.getFreeHeap();
    size_t psramFree = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    Serial.print("Free heap: ");
    Serial.println(freeHeap);
    Serial.print("Free PSRAM: ");
    Serial.println(psramFree);
}


void selectLeftChannel(){
  digitalWrite(L_R_channel_selector, LOW);
}

void selectRightChannel(){
  digitalWrite(L_R_channel_selector, HIGH);
}


void i2sInit(){

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = esp32_device.I2S_sample_rate,
    .bits_per_sample = i2s_bits_per_sample_t(esp32_device.I2S_sample_bits),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = 0,
    .dma_buf_count = 32, //try 16, 32, 48, 64, ...  --> dma_buf_len * dma_buf_count occupies heap memory
    .dma_buf_len = 1024, //at most 1024 --> dma_buf_len * dma_buf_count occupies heap memory
    .use_apll = 1
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_set_pin(I2S_PORT, &pin_config);
}




void i2sDeinit() {
    i2s_driver_uninstall(I2S_PORT);
    Serial.println("I2S Deinitialized.");
}

void wavHeader(char* i2s_read_buff){

  byte header[headerSize];

  // Set the 'RIFF' chunk descriptor
  header[0] = 'R'; // First byte of the 'RIFF' identifier
  header[1] = 'I'; // Second byte of the 'RIFF' identifier
  header[2] = 'F'; // Third byte of the 'RIFF' identifier
  header[3] = 'F'; // Fourth byte of the 'RIFF' identifier

  // Calculate the file size
  unsigned int numChannels = esp32_device.get_I2S_channel_num();
  unsigned int bitsPerSample = esp32_device.get_I2S_sample_bits();
  unsigned int sampleRate = esp32_device.get_I2S_sample_rate();
  unsigned int byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  unsigned int wavSize = esp32_device.get_audio_record_size();
  unsigned int fileSize = wavSize + headerSize - 8; // Total size of the file minus 8 bytes for 'RIFF' header

  // Set the file size in the header
  header[4] = (byte)(fileSize & 0xFF);          // Low byte of the file size
  header[5] = (byte)((fileSize >> 8) & 0xFF);   // Second byte of the file size
  header[6] = (byte)((fileSize >> 16) & 0xFF);  // Third byte of the file size
  header[7] = (byte)((fileSize >> 24) & 0xFF);  // High byte of the file size

  // Set the WAVE identifier
  header[8] = 'W';  // First byte of the 'WAVE' identifier
  header[9] = 'A';  // Second byte of the 'WAVE' identifier
  header[10] = 'V'; // Third byte of the 'WAVE' identifier
  header[11] = 'E'; // Fourth byte of the 'WAVE' identifier

  // Set the 'fmt ' subchunk header
  header[12] = 'f'; // First byte of the 'fmt ' identifier
  header[13] = 'm'; // Second byte of the 'fmt ' identifier
  header[14] = 't'; // Third byte of the 'fmt ' identifier
  header[15] = ' '; // Fourth byte of the 'fmt ' identifier

  // Set the size of the 'fmt ' chunk
  header[16] = 0x10;  // Low byte of 'fmt ' size (16 in this case)
  header[17] = 0x00;  // Second byte of 'fmt ' size (16 in this case)
  header[18] = 0x00;  // Third byte of 'fmt ' size (16 in this case)
  header[19] = 0x00;  // High byte of 'fmt ' size (16 in this case)

  // Set the audio format (1 for PCM)
  header[20] = 0x01;  // Audio format (1 for PCM)
  header[21] = 0x00;  // Reserved for high byte of audio format
  
  // Set the number of channels (1 for mono, 2 for stereo)
  header[22] = numChannels;
  header[23] = 0x00;

  // Set the sample rate (e.g., 0x00003E80==16000 for 16 KHz audio, or 0x00001F40==8000 for 8 KHz audio, or 0x00005622==22050 for 22.05 KHz audio, or 0x0000AC44==44100 for 44.1 KHz audio)
  header[24] = (byte)(sampleRate & 0xFF);         // Low byte
  header[25] = (byte)((sampleRate >> 8) & 0xFF);  // Second byte
  header[26] = (byte)((sampleRate >> 16) & 0xFF); // Third byte
  header[27] = (byte)((sampleRate >> 24) & 0xFF); // High byte

  // Set the byte rate (SampleRate * NumChannels * BitsPerSample/8)
  // 16bit-8KHz: (8000 * 1 * 16 / 8 = 16000 -> 16000 = 0x00003E80), 32bit-8KHz: (8000 * 1 * 32 / 8 = 32000 -> 32000 = 0x00007D00), 16bit-22.05KHz: (22050 * 1 * 16 / 8 = 44100 -> 44100 = 0x0000AC44)
  header[28] = (byte)(byteRate & 0xFF);           // Low byte
  header[29] = (byte)((byteRate >> 8) & 0xFF);    // Second byte
  header[30] = (byte)((byteRate >> 16) & 0xFF);   // Third byte
  header[31] = (byte)((byteRate >> 24) & 0xFF);   // High byte

  // Set block align (NumChannels * BitsPerSample/8)
  header[32] = (numChannels * bitsPerSample / 8); // Block align (32 bit sample --> 1 * 32 / 8 = 4 = 0x0004, or 16 bit sample --> 1 * 16 / 8 = 2 = 0x0002)
  header[33] = 0x00; // Reserved for high byte of block align

  // Set bits per sample (32=0x0020 for 32 bits, 16=0x0010 for 16 bits)
  header[34] = bitsPerSample; // Bits per sample (32 bits)
  header[35] = 0x00; // Reserved for high byte of bits per sample

  // Set the 'data' chunk header
  header[36] = 'd'; // First byte of 'data' identifier
  header[37] = 'a'; // Second byte of 'data' identifier
  header[38] = 't'; // Third byte of 'data' identifier
  header[39] = 'a'; // Third byte of 'data' identifier

  header[40] = (byte)(wavSize & 0xFF);          // Low byte of data size
  header[41] = (byte)((wavSize >> 8) & 0xFF);   // Second byte of data size
  header[42] = (byte)((wavSize >> 16) & 0xFF);  // Third byte of data size
  header[43] = (byte)((wavSize >> 24) & 0xFF);  // High byte of data size
  
  memcpy(i2s_read_buff, header, headerSize);

}




#define MIN_CODE_LENGTH 1
#define MAX_CODE_LENGTH 30

String generate_random_code(int length) {
  print_debug("generate_random_code");

  if (length < MIN_CODE_LENGTH || length > MAX_CODE_LENGTH) {
    Serial.println("Length must be between " + String(MIN_CODE_LENGTH) + " and " + String(MAX_CODE_LENGTH));
    return "";
  }

  String characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  String randomCode;

  for (int i = 0; i < length; i++) {
    randomCode += characters[random(0, characters.length())];  // Fixed this line
  }
  return randomCode;
}



time_t get_current_epoch(){
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return 1;
  }
  return mktime(&timeinfo);
}


String get_formatted_time_till_microseconds(){
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo, 1000)) {
    Serial.println("Failed to obtain time");
    delay(500);
    //return String(); // Return empty string in case of failure
  }

  // Format: DD-MM-YYYY_HH-MM-SS
  char formatted_time[20];
  sprintf(formatted_time, "%02d-%02d-%04d_%02d-%02d-%02d",
  timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year + 1900,
  timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);

  // Get microseconds
  long microseconds = micros() % 1000000;

  // Combine formatted time with microseconds
  char time_with_micros[30];
  sprintf(time_with_micros, "%s-%06ld", formatted_time, microseconds);

  return String(time_with_micros);

}


String get_formatted_time_till_seconds(){
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 1000)) {
    Serial.println("Failed to obtain time");
    return String(); // Return empty string in case of failure
  }

  // Format: DD-MM-YYYY_HH-MM-SS
  char formatted_time[20];
  sprintf(formatted_time, "%02d-%02d-%04d_%02d-%02d-%02d",
  timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year + 1900,
  timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);

  return String(formatted_time);

}



void i2s_record_and_notify(void *arg) {
    Serial.println("i2s_record_and_notify");
    reportMemoryStats();
    char* audio_data = (char*)arg; 
    int i2s_read_len = esp32_device.get_I2S_read_len();
    int dataSize = 0;
    size_t bytes_read;

    char* i2s_read_buff = (char*)heap_caps_calloc(i2s_read_len, sizeof(char), MALLOC_CAP_SPIRAM);
    if (!i2s_read_buff) {
        Serial.println("Failed to allocate memory for audio buffer in PSRAM");
        reportMemoryStats();
        return;
    }

    wavHeader(audio_data);
    Serial.print("Number of bytes to record: ");
    Serial.println(esp32_device.get_audio_record_size(), DEC);
    Serial.println(" *** Recording Start *** ");

    unsigned int audio_record_size = esp32_device.get_audio_record_size();
  
    while (dataSize < audio_record_size) {
    Serial.print("while (dataSize < audio_record_size) { --> "); Serial.print(dataSize);Serial.print( "<" ); Serial.println(audio_record_size);
    
    //read data from I2S bus, in this case, from ADC.
    i2s_read(I2S_PORT, (void*) i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY);

    // Check if we've read any bytes
    if (bytes_read > 0) {
      Serial.println("if (bytes_read > 0) {");
      // Ensure we don't overflow the audioData buffer
      //if (dataSize + bytes_read <= (headerSize + audio_record_size)) {
      if (dataSize <= (headerSize + audio_record_size)) {
        Serial.println("if (dataSize <= (headerSize + audio_record_size)) {");
        memcpy(audio_data + headerSize + dataSize, i2s_read_buff, bytes_read);
        //dataSize += i2s_read_len;
        dataSize += bytes_read;
        //vTaskDelay(100);
        Serial.print("Sound recording "); Serial.println(dataSize * 100 / audio_record_size);
        Serial.print("Never Used Stack Size: "); Serial.println(uxTaskGetStackHighWaterMark(NULL));
        //ets_printf("Sound recording %u%%\n", dataSize * 100 / audio_record_size);
        //ets_printf("Never Used Stack Size: %u\n", uxTaskGetStackHighWaterMark(NULL));
      } else {
        Serial.printf("Data size overflow prevented, current size: %d, max size: %d\n", dataSize + bytes_read, headerSize + audio_record_size);
      }
    }
  }

    // Update WAV header with sizes
    uint32_t riffChunkSize = 36 + dataSize;
    memcpy(audio_data + 4, &riffChunkSize, 4);
    uint32_t dataChunkSize = dataSize;
    memcpy(audio_data + 40, &dataChunkSize, 4);

    Serial.print("Size of the recorded audio data: ");
    Serial.println(dataSize);

    esp32_device.update_new_audio_data_is_available(true);



    // SAVING FILE
    record_counter += 1;
    String file_name = "/" + prefix_audio_name + "_" + String(record_counter) + ".wav";
    Serial.print("Writing audio file with name: "); Serial.println(file_name);


    i2sDeinit(); // Deinitialize I2S here

    ///////////////////////////////////////////////////////////////////////////////
    // SD-CARD INITIALIZATION
    SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);
    if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
      Serial.println("Card Mount Failed");
      return;
    }
    uint8_t cardType = SD_MMC.cardType();
    if(cardType == CARD_NONE){
      Serial.println("No SD_MMC card attached");
      return;
    }

    Serial.print("SD_MMC Card Type: ");
    if(cardType == CARD_MMC){
      Serial.println("MMC");
    } else if(cardType == CARD_SD){
      Serial.println("SDSC");
    } else if(cardType == CARD_SDHC){
      Serial.println("SDHC");
    } else {
      Serial.println("UNKNOWN");
    }

    uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
    Serial.printf("SD_MMC Card Size: %lluMB\n", cardSize);
    

    Serial.print("AUDIO_RECORD_SIZE: "); Serial.println(esp32_device.get_audio_record_size());


    writeAudioFile(SD_MMC, file_name.c_str(), audio_data, audio_data_size);
    // When done with SD card, de-initialize it properly:
    SD_MMC.end();
    ///////////////////////////////////////////////////////////////////////////////
    //i2sInit(); // Reinitialize I2S for future recordings

    // Notify that task is done
    BaseType_t result = xTaskNotifyGive(i2sTaskHandle);
    if (result != pdTRUE) {
        Serial.println("Failed to send notification");
    } else {
        Serial.println("Notification sent!");
    }

    free(i2s_read_buff);
    vTaskDelete(NULL); // Clean up task
}


// Function to deactivate WiFi
void deactivate_wifi() {
    print_debug("deactivate_wifi()");
    WiFi.disconnect();
    Serial.println("WiFi disconnected.");
}

// Function to reactivate WiFi
void activate_wifi() {
    print_debug("activate_wifi()");
    WiFi.begin(ssid_wifi, password_wifi);
    while (WiFi.status() != WL_CONNECTED) {
        Serial.println("Trying to reconnect to WiFi...\n");
        delay(500);
    }
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
    delay(1000);
}

void setup_wifi(){

	WiFi.begin(ssid_wifi, password_wifi);

  while (WiFi.status() != WL_CONNECTED) {
  	Serial.println("Trying to connect...\n");
  	delay(1000);
  }
  
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");
  //timeClient.begin();
	// Initialize NTP
	configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
}

/*

void setup() {
    Serial.begin(115200);
    reportMemoryStats();
    setup_wifi();
    reportMemoryStats();

    i2sInit();
    audio_data = (char*)heap_caps_calloc(audio_data_size, sizeof(char), MALLOC_CAP_SPIRAM);
    if (!audio_data) {
        Serial.println("Failed to allocate memory for audio data in PSRAM");
        return;
    }

    // SD-CARD INITIALIZATION
    SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);
    if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
        Serial.println("Card Mount Failed");
        return;
    }

    // Additional SD card setup and memory reporting...

    Serial.println("Setup completed");
}


*/


void setup(){
  Serial.begin(115200);
  delay(3000);
  print_debug("ENTER INTO void setup()");

  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());

  Serial.printf("Free Heap before initialize_I2S(): %d bytes\n", ESP.getFreeHeap());
  //initialize_I2S();
  Serial.printf("Free Heap before setup_wifi(): %d bytes\n", ESP.getFreeHeap());
  //setup_wifi();  //******************************************************************************************************** setup_wifi()
  Serial.printf("Free Heap before mqttClient.setup(): %d bytes\n", ESP.getFreeHeap());


  Serial.print("Total heap: ");    Serial.println(ESP.getHeapSize());
  Serial.print("Free heap: ");    Serial.println(ESP.getFreeHeap());
  Serial.print("Total PSRAM: ");    Serial.println(ESP.getPsramSize());
  Serial.print("Free PSRAM: ");    Serial.println(ESP.getFreePsram());


  if (heap_caps_get_free_size(MALLOC_CAP_SPIRAM) > 0) {
    Serial.print("PSRAM is available. Free PSRAM: ");
    Serial.println(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  } else {
    Serial.println("PSRAM is NOT available.");
  }


  // Only the left channel of the microphone will be used
  pinMode(L_R_channel_selector, OUTPUT);
  selectLeftChannel();

  
  // CHECK IF THE AUDIO RECORD SIZE IS BIGGER THAN THE PSRAM MEMORY AVAILABLE (4MB --> cut to 4'000'000 Bytes)
  if (esp32_device.audio_record_size >= 4000000){
    print_debug("if (esp32_device.audio_record_size >= 4000000){");
    print_debug("audio_record_size: "); print_debug(String(esp32_device.audio_record_size));
    unsigned int surplus_audio_size = esp32_device.audio_record_size - 4000000;
    unsigned int one_sec_audio_size = esp32_device.I2S_channel_num * esp32_device.I2S_sample_rate * esp32_device.I2S_sample_bits / 8 * 1;
    unsigned int new_audio_record_size = one_sec_audio_size;
    unsigned int num_seconds = 1;
    print_debug("new_audio_size: "); print_debug(String(new_audio_record_size));
    print_debug("num_seconds: 1");
    while (new_audio_record_size <= 4000000){
      new_audio_record_size += one_sec_audio_size;
      num_seconds += 1;
      print_debug("new_audio_size: "); print_debug(String(new_audio_record_size));
      print_debug("num_seconds: "); print_debug(String(num_seconds));
    }
    print_debug("new_audio_size: "); print_debug(String(new_audio_record_size));
    esp32_device.update_audio_record_size(new_audio_record_size);
    esp32_device.update_record_time(num_seconds);
    
  }
  esp32_device.update_I2S_read_len();
  
  Serial.println("Memory before i2sInit: ");
  reportMemoryStats();
  //i2sInit();
  Serial.println("Memory after i2sInit, before initializing audio_data: ");
  reportMemoryStats();


  // audio_data is a global variable
  audio_data = (char*)heap_caps_calloc(audio_data_size, sizeof(char), MALLOC_CAP_SPIRAM);
  if (!audio_data) {
    Serial.println("Failed to allocate memory for audio data in PSRAM");
    reportMemoryStats();
    return;
  }



  Serial.println("Memory after initializing audio_data: ");
  reportMemoryStats();

  //Waiting 4 iterations of 1 second each
  for (int t = 4; t > 0; t--) {
    //Serial.print("[SETUP] WAIT %d...\n", t);
    Serial.print("[SETUP] WAIT ");
    Serial.print(t, DEC);
    Serial.println(" ...");
    delay(500);
  }



  Serial.println("Memory before exiting from setup(): ");
  reportMemoryStats();
  print_debug("EXIT FROM void setup()");


  prefix_audio_name = generate_random_code(5);


}












time_t time_last_loop;
//String communication_id;

unsigned short int iter_loop = 0;
void loop() {
  //print_debug("void loop() {");

  
  //Serial.print("get_current_epoch() - time_last_loop = ");
  //Serial.print(get_current_epoch(),DEC);
  //Serial.print(" - ");
  //Serial.print(time_last_loop,DEC);
  //Serial.print(" = ");
  //Serial.println(get_current_epoch() - time_last_loop, DEC);

  

  if (isTaskRunning == true){
    // Check the state of the task from the loop
    eTaskState taskState = eTaskGetState(i2sTaskHandle);
    // Check the state and print or handle accordingly

    if(taskState == eRunning) {
        Serial.println("Task is running.");
    } else if(taskState == eReady) {
        Serial.println("Task is ready to run.");
    } else if(taskState == eBlocked) {
        Serial.println("Task is blocked.");
    } else if(taskState == eDeleted) {
        Serial.println("Task has been deleted.");
        isTaskRunning = false;  // already set as false at the end of the task
    } else if(taskState == eSuspended) {
        Serial.println("Task is suspended.");
    } else {
        Serial.println("Invalid task state.");
    }
  }
  
  //RECORDING AUDIO
  if (!isTaskRunning){
    // RECORDING
    // Create the I2S ADC task and store the handle
    Serial.println("Creating the task:  ");
    reportMemoryStats();
    i2sInit(); // Reinitialize I2S for future recordings

    /*
    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
      Serial.println("Failed to obtain time");
      return;
    }
    time_t now_epoch = mktime(&timeinfo);
    String formatted_time = get_formatted_time_till_microseconds();
    */

    BaseType_t result = xTaskCreate(i2s_record_and_notify, "i2s_record_and_notify", esp32_device.get_I2S_read_len(), (void*)audio_data, 2, &i2sTaskHandle);
    if (result != pdPASS) {
      Serial.println("Task creation failed");
    } else {
      Serial.println("Task created successfully");
      isTaskRunning = true;
    }
  }


  delay(1000);


    
}
