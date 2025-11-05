#include <Base64.h> // leave it here before the other libraries
#include "Arduino.h"
#include <WiFi.h>
#include <AsyncMqttClient.h>
#include "driver/i2s.h"
#include <ArduinoJson.h>
#include "device.h"
#include <time.h>
#include <esp_system.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_sleep.h"
#include <AESLib.h>
#include <esp_adc_cal.h>
#include <driver/gpio.h>
#include <driver/adc.h>

// RGB LED
//#define PIN_RED    33 //Please do not modify it
//#define PIN_GREEN  26 //Please do not modify it
//#define PIN_BLUE   27 //Please do not modify it

// ENABLES
//#define SD_CARD_3V3_ENABLE 4 //Please do not modify it
#define H2_H3_3V3_ENABLE 18 //Please do not modify it. In the header H2 is also present the 3V3 of the microphone
//#define I2C_J1_3V3_ENABLE 5 //Please do not modify it
#define BATTERY_MONITOR_ENABLE 25 //Please do not modify it

// BATTERY MONITOR ADC PORT
#define ADC_PORT_BATTERY_MONITOR 34 //Please do not modify it

// TEST MICRO SD-CARD AND MICROPHONE ENABLE
// If MICROPHONE ENABLED, MICRO SD-CARD ENABLED; --> FILE SUCCESSFULLY WRITTEN IN THE MICRO SD-CARD
// If MICROPHONE ENABLED, MICRO SD-CARD DISABLED; --> THE MICRO SD-CARD WILL NOT BE FOUND, AND THE ESP32 WILL REBOOT
// If MICROPHONE DISABLED, MICRO SD-CARD ENABLED; --> THE FILE WILL NOT BE SUCCESSFULLY WRITTEN
// If MICROPHONE DISABLED, MICRO SD-CARD DISABLED; --> THE MICRO SD-CARD WILL NOT BE FOUND, AND THE ESP32 WILL REBOOT
//#define LOGIC_SD_CARD_3V3_ENABLE_VALUE HIGH // WHEN HIGH, THE MICRO SD-CARD WILL WORK, VICEVERSA WHEN LOW
#define LOGIC_H2_H3_3V3_ENABLE_VALUE HIGH // WHEN HIGH, THE MICROPHONE WILL WORK, VICEVERSA WHEN LOW

// MICRO SD-CARD DEFINITIONS
//#define SD_MMC_CMD 15 //Please do not modify it
//#define SD_MMC_CLK 14 //Please do not modify it
//#define SD_MMC_D0  2  //Please do not modify it

// NTP Server
const char* ntpServer = "pool.ntp.org";
// Time zone offset in seconds (UTC+0), adjust as needed
const long gmtOffset_sec = 3600; // Change this based on your timezone
const int daylightOffset_sec = 3600; // Change this if you observe Daylight Saving

#define DEBUG_MODE

#define DEFAULT_VREF    1100        //Use adc2_vref_to_gpio() to obtain a better estimate
#define NO_OF_SAMPLES   128          //Multisampling

// ADC GPIO 34 PARAMETERS
static esp_adc_cal_characteristics_t *adc_chars;
#if CONFIG_IDF_TARGET_ESP32
static const adc_channel_t channel = ADC_CHANNEL_6;     //GPIO34 if ADC1, GPIO14 if ADC2
static const adc_bits_width_t width = ADC_WIDTH_BIT_12;
#elif CONFIG_IDF_TARGET_ESP32S2
static const adc_channel_t channel = ADC_CHANNEL_6;     // GPIO7 if ADC1, GPIO17 if ADC2
static const adc_bits_width_t width = ADC_WIDTH_BIT_13;
#endif
static const adc_atten_t atten = ADC_ATTEN_DB_12;
static const adc_unit_t unit = ADC_UNIT_1;

static void check_efuse(void)
{
#if CONFIG_IDF_TARGET_ESP32
    //Check if TP is burned into eFuse
    if (esp_adc_cal_check_efuse(ESP_ADC_CAL_VAL_EFUSE_TP) == ESP_OK) {
        printf("eFuse Two Point: Supported\n");
    } else {
        printf("eFuse Two Point: NOT supported\n");
    }
    //Check Vref is burned into eFuse
    if (esp_adc_cal_check_efuse(ESP_ADC_CAL_VAL_EFUSE_VREF) == ESP_OK) {
        printf("eFuse Vref: Supported\n");
    } else {
        printf("eFuse Vref: NOT supported\n");
    }
#elif CONFIG_IDF_TARGET_ESP32S2
    if (esp_adc_cal_check_efuse(ESP_ADC_CAL_VAL_EFUSE_TP) == ESP_OK) {
        printf("eFuse Two Point: Supported\n");
    } else {
        printf("Cannot retrieve eFuse Two Point calibration values. Default calibration values will be used.\n");
    }
#else
#error "This example is configured for ESP32/ESP32S2."
#endif
}

static void print_char_val_type(esp_adc_cal_value_t val_type)
{
    if (val_type == ESP_ADC_CAL_VAL_EFUSE_TP) {
        printf("Characterized using Two Point Value\n");
    } else if (val_type == ESP_ADC_CAL_VAL_EFUSE_VREF) {
        printf("Characterized using eFuse Vref\n");
    } else {
        printf("Characterized using Default Vref\n");
    }
}

// MQTT Broker details
const char* broker = "test.mosquitto.org";
const int port = 1883;
const int keepalive = 40;

// WiFi client setup
WiFiClient wifiClient;

// MQTT mqttClient setup
AsyncMqttClient mqttClient;

//Device object
Device esp32_device; // class defined in the "Device.h" header
// Global or static variable to communicate between tasks
TaskHandle_t i2sTaskHandle;
bool isTaskRunning = false;
// Define a structure to hold the topic and its QoS
struct Topic {
  const char* topic;
  int qos;
};

// Define the list of topics and QoS levels
Topic topics[] = {
  {"NodeMCU_connect_to_sensors_net/ack", 1},
  {"NodeMCU_recording_problems", 1},
  {"audio/request", 1},
  {"audio/ack", 1},
  {"NodeMCU_get_infos", 1},
};

////////////////////////// AES Encryption-Decryption functions ////////////////////////////////////////
AESLib aesLib;
byte aes_iv[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // Initialization vector buffer

// Function to encrypt a message
String encrypt_impl(char* msg, byte iv[]) {
    int msgLen = strlen(msg);
    char encrypted[2 * msgLen] = {0}; // Ensure buffer is large enough
    aesLib.encrypt64((const byte*)msg, msgLen, encrypted, aes_key, sizeof(aes_key), aes_iv);
    return String(encrypted);
}

// Function to decrypt a message
String decrypt_impl(char * msg, byte iv[]) {
  int msgLen = strlen(msg);
  char decrypted[msgLen] = {0}; // half may be enough
  aesLib.decrypt64(msg, msgLen, (byte*)decrypted, aes_key, sizeof(aes_key), iv);
  return String(decrypted);
}

////////////////////////// End of AES functions ////////////////////////////////////////

//////////////////////////I2S PARAMETERS//////////////////////////////

#define CONFIG_SPIRAM_ENABLE

//le porte GPIO corrette per l'ESP32 WROOM 32D e l'ESP32 WROVER-E sono: I2S_WS 15, I2S_SD 13, I2S_SCK 2
#define I2S_WS 15 //15
#define I2S_SCK 14 //2
#define I2S_SD 2 //13
#define I2S_PORT I2S_NUM_0
#define L_R_channel_selector 12

//The following variables have been defined as class members in device.h
//#define I2S_SAMPLE_RATE   (22050)
//#define I2S_SAMPLE_BITS   (32)
//#define RECORD_TIME       (5) //Seconds
//#define I2S_CHANNEL_NUM   (1)
//#define AUDIO_RECORD_SIZE (I2S_CHANNEL_NUM * I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 * RECORD_TIME) //5 sec 22.05KHz --> 220'500
//#define I2S_READ_LEN      (AUDIO_RECORD_SIZE / 20) // to work properly, I2S_READ_LEN must be chosen in such a way that AUDIO_RECORD_SIZE / I2S_READ_LEN equals an integer value

const int headerSize = 44;
char* audio_data;

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

/*
void i2s_record_data_scale(uint8_t * d_buff, uint8_t* s_buff, uint32_t len)
{
    uint32_t j = 0;
    uint32_t dac_value = 0;
    for (int i = 0; i < len; i += 2) {
        dac_value = ((((uint16_t) (s_buff[i + 1] & 0xf) << 8) | ((s_buff[i + 0]))));
        d_buff[j++] = 0;
        d_buff[j++] = dac_value * 256 / 2048;
    }
}
*/

void publish_audio(void *arg){
  Serial.println("publish_audio");
  //reportMemoryStats();
  char* audio_data = (char*)arg; // Cast to char*
  String dynamic_topic = "audio/payload" + esp32_device.audio_data_topic_key;
  const char* dyn_topic = dynamic_topic.c_str();

  // ENCRYPTION ////////////////////////
  aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
  byte enc_iv_X[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  String encryptedMessage = encrypt_impl((char*)dynamic_topic.c_str(), enc_iv_X);
  //////////////////////////////////////

  if (mqttClient.publish(dyn_topic, 2, false, audio_data, headerSize + esp32_device.audio_record_size)) {
    Serial.println("Published audio");
    esp32_device.new_audio_data_is_available = false; //since we sent the last audio recorded, a new audio data must be recorded ++++++++++++++++++++++++
    //reportMemoryStats();
  } else {
    Serial.println("Failed to publish audio");
    //reportMemoryStats();
  }
  //send_audio(mqttClient, dyn_topic, audio_data);
  Serial.println("Deleting task...");
  esp32_device.currently_sending_audio = false;

  vTaskDelete(NULL); // Clean up task
}

void i2s_record_and_notify(void *arg){
  Serial.println("i2s_record_and_notify");
  reportMemoryStats();  
  char* audio_data = (char*)arg; // Cast to char*

  int i2s_read_len = esp32_device.I2S_read_len;
  int dataSize = 0;
  size_t bytes_read;

  //char* i2s_read_buff = (char*) calloc(i2s_read_len, sizeof(char));
  char* i2s_read_buff = (char*)heap_caps_calloc(i2s_read_len, sizeof(char), MALLOC_CAP_SPIRAM);
  if (!i2s_read_buff) {
    Serial.println("Failed to allocate memory for audio data in PSRAM");
    reportMemoryStats();
    return;
  }
  //uint8_t* flash_write_buff = (uint8_t*) calloc(i2s_read_len, sizeof(char));
  wavHeader(audio_data);
  Serial.print("Number of bytes to record: ");
  Serial.println(esp32_device.audio_record_size, DEC);
  Serial.println(" *** Recording Start *** ");

  unsigned int audio_record_size = esp32_device.audio_record_size;

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
        vTaskDelay(100);
        Serial.print("Sound recording "); Serial.println(dataSize * 100 / audio_record_size);
        Serial.print("Never Used Stack Size: "); Serial.println(uxTaskGetStackHighWaterMark(NULL));
        //ets_printf("Sound recording %u%%\n", dataSize * 100 / audio_record_size);
        //ets_printf("Never Used Stack Size: %u\n", uxTaskGetStackHighWaterMark(NULL));
      } else {
        Serial.printf("Data size overflow prevented, current size: %d, max size: %d\n", dataSize + bytes_read, headerSize + audio_record_size);
      }
    }
  }

  // Update the WAV header with correct sizes
  uint32_t riffChunkSize = 36 + dataSize;
  memcpy(audio_data + 4, &riffChunkSize, 4); // Write RIFF chunk size
  
  uint32_t dataChunkSize = dataSize;
  memcpy(audio_data + 40, &dataChunkSize, 4); // Write Data chunk size

  Serial.print("Size of the recorded audio data: ");
  Serial.println(dataSize);

  //free(buffer); // Free the buffer allocated in PSRAM
  Serial.println("Recording finished");
  esp32_device.new_audio_data_is_available = true;
  Serial.print("Size of the recorded audio data: ");
  Serial.println(dataSize); // Print the number of bytes actually stored

  i2sDeinit();

  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return;
  }
  time_t now_epoch = mktime(&timeinfo);
  
  String formatted_time = get_formatted_time_till_microseconds();
  Serial.println("A formatted_time: " + formatted_time);
  esp32_device.last_record_epoch = now_epoch;
  esp32_device.last_record_date = formatted_time;

  /*
  if (mqttClient.publish("receive_audio_test", 2, false, (const char*)audio_data, headerSize + AUDIO_RECORD_SIZE)) {
    Serial.println("Published audio");
  }
  */

  //free(flash_write_buff);
  //flash_write_buff = NULL;
  
  //listSPIFFS();

  //Notify the raspberry that the audio is available and ready to be sent
  publish_infos_with_communication_id(mqttClient, "Raspberry_audio_available", esp32_device, generate_random_code(20));
  delay(2000);  //just to be sure the raspberry receives this message

  // Notify that task is done
  BaseType_t result = xTaskNotifyGive(i2sTaskHandle);
  if (result != pdTRUE) {
    Serial.println("Failed to send notification");
  } else{
    Serial.println("Notification sent!");
  }
  
  free(i2s_read_buff);
  //i2s_read_buff = NULL;

  Serial.println("Exit from i2s_record_and_notify");
  reportMemoryStats();
  isTaskRunning = false;
  Serial.println("Deleting task...");
  vTaskDelete(NULL); // Clean up task
}

void wavHeader(char* i2s_read_buff){

  byte header[headerSize];

  // Set the 'RIFF' chunk descriptor
  header[0] = 'R'; // First byte of the 'RIFF' identifier
  header[1] = 'I'; // Second byte of the 'RIFF' identifier
  header[2] = 'F'; // Third byte of the 'RIFF' identifier
  header[3] = 'F'; // Fourth byte of the 'RIFF' identifier

  // Calculate the file size
  unsigned int numChannels = esp32_device.I2S_channel_num;
  unsigned int bitsPerSample = esp32_device.I2S_sample_bits;
  unsigned int sampleRate = esp32_device.I2S_sample_rate;
  unsigned int byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  unsigned int wavSize = esp32_device.audio_record_size;
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

///////////////////////////END OF I2S PARAMETERS///////////////////////////

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
  if (!getLocalTime(&timeinfo, 1000)) {
    Serial.println("Failed to obtain time");
    return String(); // Return empty string in case of failure
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



float get_Battery_Voltage(){

  //set_rgb_led_color(0, 201, 204);

  delay(1000); // keep the color 1 second

  //Check if Two Point or Vref are burned into eFuse
  check_efuse();

  //Configure ADC
  if (unit == ADC_UNIT_1) {
      adc1_config_width(width);
      adc1_config_channel_atten((adc1_channel_t)channel, atten);
  } else {
      adc2_config_channel_atten((adc2_channel_t)channel, atten);
  }

  //Characterize ADC
  adc_chars = (esp_adc_cal_characteristics_t *)calloc(1, sizeof(esp_adc_cal_characteristics_t));
  esp_adc_cal_value_t val_type = esp_adc_cal_characterize(unit, atten, width, DEFAULT_VREF, adc_chars);
  print_char_val_type(val_type);

  //Continuously sample ADC1
  uint32_t adc_reading = 0;
  uint32_t adc_reading_avg = 0;
  float battery_voltage_mV = 0.0;

  //Multisampling
  for (int i = 0; i < NO_OF_SAMPLES; i++) {
      if (unit == ADC_UNIT_1) {
          adc_reading = adc1_get_raw((adc1_channel_t)channel);
          printf("adc_reading: %d\n", adc_reading);
          adc_reading_avg += adc_reading;
      } else { // nel nostro caso non avviene mai, perchè viene usato sempre GPIO 34 per misurare half_voltage della batteria
          int raw;
          adc2_get_raw((adc2_channel_t)channel, width, &raw);
          adc_reading_avg += raw;
      }
  }
  adc_reading_avg /= NO_OF_SAMPLES;

  // nel caso la tensione della batteria uscisse dal range di tensione di ADC_ATTEN_DB_12, allora usiamo ADC_ATTEN_DB_6, 
  // che è per il range di tensione subito inferiore a quello di ADC_ATTEN_DB_12
  if (adc_reading_avg >= 4090){
    printf("if (adc_reading_avg >= 4090)\n");
    //Check if Two Point or Vref are burned into eFuse
    check_efuse();

    //Configure ADC
    if (unit == ADC_UNIT_1) {
        adc1_config_width(width);
        adc1_config_channel_atten((adc1_channel_t)channel, ADC_ATTEN_DB_6);
    } else {
        adc2_config_channel_atten((adc2_channel_t)channel, ADC_ATTEN_DB_6);
    }

    //Characterize ADC
    adc_chars = (esp_adc_cal_characteristics_t *)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_value_t val_type = esp_adc_cal_characterize(unit, ADC_ATTEN_DB_6, width, DEFAULT_VREF, adc_chars);
    print_char_val_type(val_type);

    //Continuously sample ADC1
    uint32_t adc_reading = 0;
    uint32_t adc_reading_avg = 0;
    //Multisampling
    for (int i = 0; i < NO_OF_SAMPLES; i++) {
        if (unit == ADC_UNIT_1) {
            adc_reading = adc1_get_raw((adc1_channel_t)channel);
            printf("adc_reading: %d\n", adc_reading);
            adc_reading_avg += adc_reading;
        } else { // nel nostro caso non avviene mai, perchè viene usato sempre GPIO 34 per misurare half_voltage della batteria
            int raw;
            adc2_get_raw((adc2_channel_t)channel, width, &raw);
            adc_reading_avg += raw;
        }
    }
    adc_reading_avg /= NO_OF_SAMPLES;

    if (adc_reading_avg >= 4090 || adc_reading_avg <=1200){
      printf("\n\n\n\n- - - - - - - - - THE BATTERY IS NOT CONNECTED - - - - - - - - - -\n\n\n\n");
    }

  }else if (adc_reading_avg <=1200){
    printf("\n\n\n\n- - - - - - - - - THE BATTERY IS NOT CONNECTED - - - - - - - - - -\n\n\n\n");
  }

  //Convert adc_reading to voltage in mV
  printf("adc_reading_avg: %d\n", adc_reading_avg);
  uint32_t half_voltage = esp_adc_cal_raw_to_voltage(adc_reading_avg, adc_chars);
  float voltage = 2*(float)half_voltage;
  printf("half_voltage: %d\n", half_voltage);
  printf("voltage: %f\n", voltage);

  esp32_device.update_battery_level((float)(voltage)/1000);

  printf("\n _______________________________________________________\n");
  printf("|                                                       |\n");
  printf("|                                                       |\n");
  printf("|                                                       |\n");
  printf("Raw: %d\tVoltage: %fmV\n", adc_reading_avg, voltage-60); // 60 mV vengono tolti perchè l'ADC registra 30mV in più (moltiplicati per 2 diventano 60)
  //printf("esp32_device.battery_level: %fV\n", voltage);
  printf("|                                                       |\n");
  printf("|                                                       |\n");
  printf("|                                                       |\n");
  printf(" -------------------------------------------------------\n");

  //vTaskDelay(pdMS_TO_TICKS(1000));

  digitalWrite(BATTERY_MONITOR_ENABLE, LOW);
  //turn_off_rgb_led();
  battery_voltage_mV = float((voltage-60)/1000);

  return battery_voltage_mV;
}






void publish_infos_with_communication_id(AsyncMqttClient& mqttClient, const String& topic, Device& esp32_device, const String& communication_id) {
  print_debug("publish_infos_with_communication_id");
  DynamicJsonDocument messageDoc(512);
  // Create the JSON message
  messageDoc["dummy-pad"] = ""; // leave it like this: messageDoc["dummy-pad"] = "";
  messageDoc["infos"] = esp32_device.all_infos_to_dict();
  messageDoc["communication_id"] = communication_id;
  String message;
  serializeJson(messageDoc, message);
	String formatted_time = get_formatted_time_till_seconds();
	Serial.println("B formatted_time: " + formatted_time);
  esp32_device.communication_id = communication_id;
  print_debug("+++++++++++++++++++++++++++++");
  print_debug(("formatted_time: " + formatted_time).c_str());
  print_debug(("topic: " + topic).c_str());
  print_debug(("Sending: " + message).c_str());

  // ENCRYPTION ////////////////////////
  aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
  byte enc_iv_X[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  String encryptedMessage = encrypt_impl((char*)message.c_str(), enc_iv_X);
  //////////////////////////////////////

  mqttClient.publish(topic.c_str(), 2, false, encryptedMessage.c_str(), encryptedMessage.length()); // Use retain true if needed
}

void publish_infos_with_invitation_id(AsyncMqttClient& mqttClient, const String& topic, Device& esp32_device, const String& invitation_id) {
  print_debug("publish_infos_with_invitation_id");
  DynamicJsonDocument messageDoc(512);
  // Create the JSON message
  messageDoc["dummy-pad"] = ""; // leave it like this: messageDoc["dummy-pad"] = "";
  messageDoc["infos"] = esp32_device.all_infos_to_dict();
  messageDoc["invitation_id"] = invitation_id;
  messageDoc["net_password"] = esp32_device.net_password; // esp32_device.net_password;
  String message;
  serializeJson(messageDoc, message);
  esp32_device.invitation_id = invitation_id;
	String formatted_time = get_formatted_time_till_seconds();
	Serial.println("C formatted_time: " + formatted_time);
  print_debug("+++++++++++++++++++++++++++++");
  print_debug(("formatted_time: " + formatted_time).c_str());
  print_debug(("topic: " + topic).c_str());
  print_debug(("Sending: " + message).c_str());

  // ENCRYPTION ////////////////////////
  aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
  byte enc_iv_X[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  String encryptedMessage = encrypt_impl((char*)message.c_str(), enc_iv_X);
  //////////////////////////////////////

  mqttClient.publish(topic.c_str(), 2, false, encryptedMessage.c_str(), encryptedMessage.length()); // Use retain if needed
}

// Callback function for handling received messages

// Very important: As a rule of thumb, never use blocking functions in the callbacks (don't use delay() or yield()). Otherwise, you may very probably experience unexpected behaviors.
// void onMessage(char* topic, uint8_t* payload, size_t length, AsyncMqttClientMessageProperties properties, size_t index, size_t total)
//void on_message(char* topic, byte* payload, unsigned int length) {
void on_message(char* topic, char* payload, AsyncMqttClientMessageProperties properties, unsigned int length, unsigned int packetId, unsigned int messageId){
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(" with length: ");
  Serial.println(length);
  
  payload[length] = '\0'; // Ensure null-termination for string functions
  String message = String((char*)payload);
 
  Serial.print("Message: ");
  Serial.println(message);
    
  if (strcmp(topic, "NodeMCU_connect_to_sensors_net/ack") == 0) {
    DynamicJsonDocument doc(1024);
    String formatted_time = get_formatted_time_till_seconds();
		print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    print_debug("NodeMCU_connect_to_sensors_net/ack");
    Serial.println("D formatted_time: " + formatted_time);

    // DECRYPTION ///////////////
    aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
    byte enc_iv_Y[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    String decryptedMessage = decrypt_impl((char*)message.c_str(), enc_iv_Y);
    if (decryptedMessage.length() > 16) {
      decryptedMessage = decryptedMessage.substring(16);  //get rid of the first 16 garbage characters
    }
    Serial.print("decryptedMessage: ");
    Serial.println(decryptedMessage);
    /////////////////////////////

    try {
      if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok) {
        Serial.print("if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok)");
        if (doc["invitation_id"] == esp32_device.invitation_id) { //+++++++++++++++++++++++++++++++
          print_debug("if (doc['invitation_id'] == esp32_device.invitation_id) {");
          if (doc["Connection_permission"] == "allowed" || doc["Connection_permission"] == "already_in") {
            print_debug("if (doc['Connection_permission'] == 'allowed' || doc['Connection_permission'] == 'already_in') {");
            
            esp32_device.device_id = doc["device_id"].as<String>(); // Update device ID
            esp32_device.subscribed_flag = true; // Device successfully subscribed +++++++++++++++++++++++++++
            int current_raspberrypi_timer = doc["current_timer"];
            int deep_sleep_time = doc["deep_sleep_time"];
            Serial.print("current_raspberrypi_timer: "); Serial.print(current_raspberrypi_timer);
            Serial.print(" and deep_sleep_time: "); Serial.println(deep_sleep_time);

            int seconds_before_audio_request = 1;
            if (deep_sleep_time - current_raspberrypi_timer > seconds_before_audio_request){
              Serial.println("if (deep_sleep_time - current_raspberrypi_timer > seconds_before_audio_request)");
              int time_to_sleep = deep_sleep_time - current_raspberrypi_timer - seconds_before_audio_request;
              Serial.println("time_to_sleep: " + time_to_sleep);
              publish_infos_with_invitation_id(mqttClient, "Raspberry_connect_to_sensors_net/ack", esp32_device, doc["invitation_id"]);
              esp32_device.deep_sleep_current_time = time_to_sleep;
              esp32_device.deep_sleep_flag = true;
              
              //delay(1000);
              //esp_sleep_enable_timer_wakeup(time_to_sleep * 1000000); // Convert seconds to microseconds
              // Now go to deep sleep
              //Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(time_to_sleep); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
              //esp_deep_sleep_start();

            } else {
              Serial.println("else if (deep_sleep_time - current_raspberrypi_timer <= seconds_before_audio_request)");
              publish_infos_with_invitation_id(mqttClient, "Raspberry_connect_to_sensors_net/ack", esp32_device, doc["invitation_id"]);
              esp32_device.start_recording_now = true; //+++++++++++++++++++
              esp32_device.deep_sleep_time = doc["deep_sleep_time"];
              esp32_device.max_time_on = doc["max_time_on"];
              esp32_device.I2S_sample_rate = doc["I2S_sample_rate"];
              esp32_device.record_time = doc["record_time"];
              esp32_device.I2S_channel_num = doc["I2S_channel_num"];
              esp32_device.I2S_sample_bits = doc["I2S_sample_bits"];
              esp32_device.update_audio_record_size();
              esp32_device.update_I2S_read_len();
              esp32_device.ready_to_set_device_flag = true;
            }

          }else if (doc["Connection_permission"] == "denied") {
            print_debug("else if (doc['Connection_permission'] == 'denied') {");
            esp32_device.subscribed_flag = false;
            //esp32_device.pending_subscription = true;
            esp32_device.deep_sleep_time = doc["deep_sleep_time"];
            esp32_device.max_time_on = doc["max_time_on"];
            
            esp32_device.deep_sleep_current_time = esp32_device.deep_sleep_time;
            esp32_device.deep_sleep_flag = true;
            // Set up the timer wakeup
            //esp_sleep_enable_timer_wakeup(esp32_device.deep_sleep_time * 1000000); // Convert seconds to microseconds
            //Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(esp32_device.deep_sleep_time); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
            // Now go to deep sleep
            //esp_deep_sleep_start();
          }
        }
      } else {
        Serial.println("Caught an exception while processing the JSON message for the topic 'NodeMCU_connect_to_sensors_net/ack'.");
      }
    }catch (...) {
      Serial.println("Caught an exception while processing the JSON message for the topic 'NodeMCU_connect_to_sensors_net/ack'.");
    }
	  Serial.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
	}

  else if (strcmp(topic, "NodeMCU_recording_problems") == 0) {
    DynamicJsonDocument doc(1024);
    String formatted_time = get_formatted_time_till_seconds();
		print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    print_debug("NodeMCU_recording_problems");
		Serial.println("E formatted_time: " + formatted_time);

    // DECRYPTION ///////////////
    aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
    byte enc_iv_Y[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    String decryptedMessage = decrypt_impl((char*)message.c_str(), enc_iv_Y);
    if (decryptedMessage.length() > 16) {
      decryptedMessage = decryptedMessage.substring(16);  //get rid of the first 16 garbage characters
    }
    Serial.print("decryptedMessage: ");
    Serial.println(decryptedMessage);
    /////////////////////////////

    try {
		  if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok) {
		    if (doc["communication_id"] == esp32_device.communication_id){
					esp32_device.enable_audio_record = false;
				}
			}else{
        Serial.println("Caught an exception while processing the JSON message for the topic 'audio/request'.");
      }
		}catch (...) {
			Serial.println("Caught an exception while processing the JSON message for the topic 'NodeMCU_recording_problems'.");
		}
		Serial.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
	}

  else if (strcmp(topic, "audio/request") == 0) {
    DynamicJsonDocument doc(1024);
    String formatted_time = get_formatted_time_till_seconds();
		print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    print_debug("audio/request");
		Serial.println("F formatted_time: " + formatted_time);

    // DECRYPTION ///////////////
    aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
    byte enc_iv_Y[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    String decryptedMessage = decrypt_impl((char*)message.c_str(), enc_iv_Y);
    if (decryptedMessage.length() > 16) {
      decryptedMessage = decryptedMessage.substring(16);  //get rid of the first 16 garbage characters
    }
    Serial.print("decryptedMessage: ");
    Serial.println(decryptedMessage);
    /////////////////////////////

    try {
		  if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok) {
		    if (doc["device_id"] == esp32_device.device_id){
		      esp32_device.communication_id = doc["communication_id"].as<String>();
		      esp32_device.audio_data_topic_key = doc["audio_data_topic_key"].as<String>();
          esp32_device.start_sending_audio_now = true;
				}
			}else{
        Serial.println("Caught an exception while processing the JSON message for the topic 'audio/request'.");
      }
		}catch (...) {
			Serial.println("Caught an exception while processing the JSON message for the topic 'audio/request'.");
		}
		Serial.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
	}

  else if (strcmp(topic, "audio/ack") == 0) {
    DynamicJsonDocument doc(1024);
    String formatted_time = get_formatted_time_till_seconds();
		print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    print_debug("audio/ack");
		Serial.println("G formatted_time: " + formatted_time);

    // DECRYPTION ///////////////
    aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
    byte enc_iv_Y[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    String decryptedMessage = decrypt_impl((char*)message.c_str(), enc_iv_Y);
    if (decryptedMessage.length() > 16) {
      decryptedMessage = decryptedMessage.substring(16);  //get rid of the first 16 garbage characters
    }
    Serial.print("decryptedMessage: ");
    Serial.println(decryptedMessage);
    /////////////////////////////

    try {
		  if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok) {
		    if (doc["communication_id"] == esp32_device.communication_id){
		      if (doc["ack"] == "ok"){
		        print_debug("if (doc['ack'] == 'ok'){");
            // Set up the timer wakeup
            esp32_device.deep_sleep_current_time = esp32_device.deep_sleep_time;
            esp32_device.deep_sleep_flag = true;

            //esp_sleep_enable_timer_wakeup(esp32_device.deep_sleep_time * 1000000); // Convert seconds to microseconds
            // Now go to deep sleep
            //Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(esp32_device.deep_sleep_time); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
            //esp_deep_sleep_start();
		      } else if (doc["ack"] == "resend"){
		        print_debug("else if (doc['ack'] == 'resend'){");
		        esp32_device.communication_id = doc["communication_id"].as<String>();
		        esp32_device.audio_data_topic_key = doc["audio_data_topic_key"].as<String>();
            esp32_device.start_sending_audio_now = true;
				  }
			  }
      }else{
        Serial.println("Caught an exception while processing the JSON message for the topic 'audio/request'.");
      }
		}catch (...) {
			Serial.println("Caught an exception while processing the JSON message for the topic 'audio/ack'.");
		}
	  Serial.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
	}

  else if (strcmp(topic, "NodeMCU_get_infos") == 0) {
    DynamicJsonDocument doc(1024);
    String formatted_time = get_formatted_time_till_seconds();
		print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    print_debug("NodeMCU_get_infos");
		Serial.println("H formatted_time: " + formatted_time);

    // DECRYPTION ///////////////
    aesLib.set_paddingmode(paddingMode::CMS); // Using CMS padding
    byte enc_iv_Y[N_BLOCK] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    String decryptedMessage = decrypt_impl((char*)message.c_str(), enc_iv_Y);
    if (decryptedMessage.length() > 16) {
      decryptedMessage = decryptedMessage.substring(16);  //get rid of the first 16 garbage characters
    }
    Serial.print("decryptedMessage: ");
    Serial.println(decryptedMessage);
    /////////////////////////////

    try {
		  if (deserializeJson(doc, decryptedMessage) == DeserializationError::Ok) {
		    if (doc["device_id"] == esp32_device.device_id){
		      esp32_device.communication_id = doc["communication_id"].as<String>();
		      publish_infos_with_communication_id(mqttClient, "Raspberry_get_infos", esp32_device, doc["communication_id"]);
				}
			}else{
        Serial.println("Caught an exception while processing the JSON message for the topic 'audio/request'.");
      }
		}catch (...) {
			Serial.println("Caught an exception while processing the JSON message for the topic 'NodeMCU_get_infos'.");
		}
		Serial.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
	}

  String formatted_time = get_formatted_time_till_seconds();
	print_debug("OUT from on_message");
	Serial.println("I formatted_time: " + formatted_time);

} //end of the on_message callback function

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
// Function to reconnect the MQTT mqttClient
void reconnect() {
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (mqttClient.connect("ESP32_Client_ID")) { // Use Unique Client ID
      Serial.println(" connected");
      // Subscribe to all topics
      for (const char* topic : topics) {          
        mqttClient.subscribe(topic);
        Serial.printf("Subscribed to topic: %s\n", topic);
      }
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      delay(2000);
    }
  }
}
*/

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

// Number of topics
const int numTopics = sizeof(topics) / sizeof(topics[0]);

void on_connect(bool sessionPresent) {
  // Subscribe to topics
  for (int i = 0; i < numTopics; i++) {
    if (mqttClient.subscribe(topics[i].topic, topics[i].qos)) {
      Serial.print("Subscribed to: ");
      Serial.println(topics[i].topic);
    } else {
      Serial.print("Failed to subscribe to: ");
      Serial.println(topics[i].topic);
    }
  }
}


void setup(){
  Serial.begin(115200);
  
  print_debug("ENTER INTO void setup()");

  Serial.println("ENTER INTO void setup()");
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());

  Serial.printf("Free Heap before initialize_I2S(): %d bytes\n", ESP.getFreeHeap());
  //initialize_I2S();
  Serial.printf("Free Heap before setup_wifi(): %d bytes\n", ESP.getFreeHeap());
  setup_wifi();
  Serial.printf("Free Heap before mqttClient.setup(): %d bytes\n", ESP.getFreeHeap());

	// Set keep-alive interval (in seconds)
  mqttClient.setKeepAlive(keepalive); // Set keep-alive interval in seconds

  // Set up MQTT callbacks
  mqttClient.onConnect(on_connect);
  mqttClient.onMessage(on_message);
  	
  // Connect to the MQTT broker with a keep-alive interval
  mqttClient.setServer(broker, port);
  //const char* client_id = (String("ESP32ClientID") + generate_random_code(10)).c_str();
	//mqttClient.setClientId(client_id);
  //mqttClient.setProtocolVersion(MQTT_VERSION_5); // Set to MQTT v5
  mqttClient.connect();

  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT...");
    delay(2000);
    // Use the connect method with the Client ID and keep-alive interval
    /*
    if (mqttClient.connect()) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      delay(2000);
    }
    */
  }

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

  pinMode(H2_H3_3V3_ENABLE, OUTPUT);
  pinMode(BATTERY_MONITOR_ENABLE, OUTPUT);
  digitalWrite(BATTERY_MONITOR_ENABLE, HIGH);
  Serial.printf("\n\nWaiting %d seconds before checking the battery voltage . . . .\n\n", 2);
  delay(2000);
  esp32_device.battery_level = get_Battery_Voltage();
  printf("Battery Voltage in mV: %f\n", esp32_device.battery_level);
  String str_battery_voltage_V = String(esp32_device.battery_level,2);
  Serial.println("str_battery_voltage_V: " + str_battery_voltage_V);
  digitalWrite(BATTERY_MONITOR_ENABLE, LOW);

  // Get the IP address
  IPAddress ip = WiFi.localIP();
  // Convert IPAddress to String
  String ipString = ip.toString();
  // Update IP address
  esp32_device.ip_address = ipString;
  // Print IP address
  Serial.println("IP Address: " + esp32_device.ip_address);



  esp32_device.net_password = net_password; // password of the sensors network, configured in credentials.h
	// Start asking to be part of the sensors network
	publish_infos_with_invitation_id(mqttClient, "Raspberry_connect_to_sensors_net/request", esp32_device, generate_random_code(20));
  esp32_device.epoch_time_at_subscription = get_current_epoch(); // epoch time at the first subscription attempt
  Serial.print("esp32_device.invitation_id: ");
  Serial.println(esp32_device.invitation_id);

  //go on with settings only if the the device is successfully subscribed and when it should not go in deep sleep
  Serial.println("while(esp32_device.ready_to_set_device_flag == false and esp32_device.deep_sleep_flag == false)");
  while(esp32_device.ready_to_set_device_flag == false and esp32_device.deep_sleep_flag == false){
    delay(500);
  }

  int audio_data_size = headerSize + esp32_device.audio_record_size;
  unsigned int AUDIO_RECORD_SIZE = esp32_device.audio_record_size;

  // audio_data is a global variable
  audio_data = (char*)heap_caps_calloc(audio_data_size, sizeof(char), MALLOC_CAP_SPIRAM);
  if (!audio_data) {
    Serial.println("Failed to allocate memory for audio data in PSRAM");
    reportMemoryStats();
    return;
  }



  Serial.println("Memory before exiting from setup(): ");
  reportMemoryStats();
  print_debug("EXIT FROM void setup()");

}

time_t time_last_loop;
//String communication_id;

unsigned short int iter_loop = 0;
void loop() {
  print_debug("void loop() {");

  if (esp32_device.deep_sleep_flag == true){
    Serial.println("if (esp32_device.deep_sleep_flag == true)");
    esp_sleep_enable_timer_wakeup(esp32_device.deep_sleep_current_time * 1000000); // Convert seconds to microseconds
    // Now go to deep sleep
    Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(esp32_device.deep_sleep_current_time); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
    esp_deep_sleep_start();
  }
  
  //Serial.print("get_current_epoch() - time_last_loop = ");
  //Serial.print(get_current_epoch(),DEC);
  //Serial.print(" - ");
  //Serial.print(time_last_loop,DEC);
  //Serial.print(" = ");
  //Serial.println(get_current_epoch() - time_last_loop, DEC);


	if(get_current_epoch() - time_last_loop >= 1){
    print_debug("if(get_current_epoch() - time_last_loop >= 1){");
    time_last_loop = get_current_epoch();
    iter_loop +=1;
    if (iter_loop%100 == 0){
      yield();
    }

    if(esp32_device.subscribed_flag == true){
      Serial.print("if(esp32_device.subscribed_flag == true){");
      if(get_current_epoch() - esp32_device.epoch_time_at_subscription >= esp32_device.max_time_on){
        // Set up the timer wakeup
        Serial.print("THE DEVICE IS ON FOR TOO LONG: "); Serial.print(get_current_epoch() - esp32_device.epoch_time_at_subscription); Serial.println(" seconds.");
        unsigned int sleep_time = 1; // in seconds
        esp_sleep_enable_timer_wakeup(sleep_time * 1000000); // Convert seconds to microseconds
        Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(sleep_time); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
        // Now go to deep sleep
        esp_deep_sleep_start();
      }
    }
    
    if(esp32_device.subscribed_flag == false){
      Serial.print("if(esp32_device.subscribed_flag == false){");
      if(get_current_epoch() - esp32_device.epoch_time_at_subscription >= esp32_device.max_time_for_successful_subscription){
        // Set up the timer wakeup
        Serial.print("THE DEVICE IS ON BUT NOT SUBSCRIBED FOR TOO LONG: "); Serial.print(get_current_epoch() - esp32_device.epoch_time_at_subscription); Serial.println(" seconds.");
        unsigned int sleep_time = 40; // in seconds
        esp_sleep_enable_timer_wakeup(sleep_time * 1000000); // Convert seconds to microseconds
        Serial.print("!!!!!!!!!!!!!!!!!  Going to sleep for: "); Serial.print(sleep_time, DEC); Serial.println(" seconds!!!!!!!!!!!!!!!!!!!!");
        // Now go to deep sleep
        esp_deep_sleep_start();
      }
    }

		// Ensure the mqttClient stays connected and processes incoming messages
		if (!mqttClient.connected()) {
		  // Reconnect logic here
		  while (!mqttClient.connected()) {
		    Serial.print("Reconnecting to MQTT...");
        delay(2000);
        /*
		    if (mqttClient.connect()) {
		      Serial.println("reconnected");
		      // Resubscribe to the topics as needed
		      for (int i = 0; i < numTopics; i++) {
		        mqttClient.subscribe(topics[i].topic, topics[i].qos);
		      }
		    } else {
		      Serial.print("failed, rc=");
		      Serial.print(mqttClient.state());
		      delay(2000);
		    }
        */
		  }
		}

		//mqttClient.loop(); // Process incoming messages

    if(WiFi.status() == WL_CONNECTED){
		  // CODICE PRINCIPALE QUI
      print_debug("if(WiFi.status() == WL_CONNECTED){");
      print_debug("esp32_device.subscribed_flag: ");
      Serial.println(esp32_device.subscribed_flag);
      print_debug("esp32_device.start_recording_now: ");
      Serial.println(esp32_device.start_recording_now);
      print_debug("esp32_device.start_sending_audio_now: ");
      Serial.println(esp32_device.start_sending_audio_now);
      print_debug("esp32_device.currently_sending_audio: ");
      Serial.println(esp32_device.currently_sending_audio);
      print_debug("esp32_device.new_audio_data_is_available: ");
      Serial.println(esp32_device.new_audio_data_is_available);
      print_debug("isTaskRunning: ");
      Serial.println(isTaskRunning);
     
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

      //SENDING AUDIO
      if (esp32_device.start_sending_audio_now == true) {
        print_debug("if (esp32_device.start_sending_audio_now == true) {");
        String formatted_time = get_formatted_time_till_seconds();
		    Serial.println("J formatted_time: " + formatted_time);
        String dynamic_topic = "audio/payload" + esp32_device.audio_data_topic_key;
        //publish_wav_file(mqttClient, dynamic_topic, file_path);
        // Wait for notification that recording has completed

        if (isTaskRunning == true) {
          //Serial.println("if (i2sTaskHandle) {");
          // Wait for notification from the I2S task that recording is done
          Serial.print("portMAX_DELAY = ");
          Serial.println(portMAX_DELAY, DEC);
          Serial.print("configTICK_RATE_HZ = ");
          Serial.println(configTICK_RATE_HZ, DEC);
          // Example of taking a notification
          Serial.println("Waiting for the recording task to finish");
          String formatted_time = get_formatted_time_till_microseconds();
          Serial.println("K formatted_time: " + formatted_time);
          if (ulTaskNotifyTake(pdTRUE, 22000) == 0) { // Wait for 22 seconds
              Serial.println("Notification Failed or timed out.");
              esp32_device.start_sending_audio_now = false;
          } else {
              Serial.println("Notification received!");
          }
          Serial.println("End of waiting at:");
          formatted_time = get_formatted_time_till_microseconds();
          Serial.println("L formatted_time: " + formatted_time);

          // Once the notification is received, we can safely delete the task
          if (esp32_device.start_sending_audio_now == true){
            //Send audio
            Serial.println("+++++++++++++++++++++++++");
            Serial.println("Sending audio");
            reportMemoryStats();
            if (audio_data == nullptr) {
              Serial.println("audio_data is null, cannot publish");
              return;
            }
            esp32_device.currently_sending_audio = true;
            BaseType_t result = xTaskCreate(publish_audio, "publish_audio", 1024 * 2, (void*)audio_data, 2, NULL);
            if (result != pdPASS) {
              Serial.println("Task creation failed");
              esp32_device.currently_sending_audio = false;
            } else {
              Serial.println("Task created successfully");
            }
            
            Serial.println("before free");
            free(audio_data);
            audio_data = NULL;
            Serial.println("after free");
          }
          //esp32_device.start_recording_now = true;
          esp32_device.start_recording_now = false; // no more recordings, the device should go on deep sleep mode***********************************
          esp32_device.start_sending_audio_now = false;
        } else {
          //Send audio
          Serial.println("+++++++++++++++++++++++++");
          Serial.println("Sending audio");
          reportMemoryStats();

          if (audio_data == nullptr) {
            Serial.println("audio_data is null, cannot publish");
            return;
          }

          esp32_device.currently_sending_audio = true;
          BaseType_t result = xTaskCreate(publish_audio, "publish_audio", 1024 * 2, (void*)audio_data, 2, NULL);
          if (result != pdPASS) {
            Serial.println("Task creation failed");
            esp32_device.currently_sending_audio = false;
          } else {
            Serial.println("Task created successfully");
          }

          //esp32_device.start_recording_now = true;
          esp32_device.start_recording_now = false; // no more recordings, the device should go on deep sleep mode***********************************
          esp32_device.start_sending_audio_now = false;
        }
        
      }

      //RECORDING AUDIO
      if (esp32_device.start_recording_now == true && esp32_device.currently_sending_audio == false && esp32_device.new_audio_data_is_available == false){
        Serial.println("if (esp32_device.start_recording_now == true && esp32_device.currently_sending_audio == false){");
        esp32_device.start_recording_now = false;
        /*
        audio_data = (char*)heap_caps_calloc(audio_data_size, sizeof(char), MALLOC_CAP_SPIRAM);
        if (!audio_data) {
          Serial.println("Failed to allocate memory for audio data in PSRAM");
          reportMemoryStats();
          return;
        }
        */
        if (!isTaskRunning){
          // RECORDING********************************
          // Create the I2S ADC task and store the handle
          Serial.println("Creating the task:  ");

          i2sInit(); // Reinitialize I2S for future recordings
          digitalWrite(H2_H3_3V3_ENABLE, LOGIC_H2_H3_3V3_ENABLE_VALUE); // enable H2(so also the microphone) and H3

          delay(3000);          

          BaseType_t result = xTaskCreate(i2s_record_and_notify, "i2s_record_and_notify", esp32_device.I2S_read_len, (void*)audio_data, 2, &i2sTaskHandle);
          if (result != pdPASS) {
            Serial.println("Task creation failed");
          } else {
            Serial.println("Task created successfully");
            isTaskRunning = true;
          }
        }

      }

      delay(1000);

    } // if(WiFi.status() == WL_CONNECTED)

  } // if(get_current_epoch() - time_last_loop >= 500)
    
}
