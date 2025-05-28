#include <Arduino.h> // Ensure you have this if you're using Arduino
#include <ArduinoJson.h>
#include <string>
#include "credentials.h"
#define DEBUG_MODE


// Function to print debug messages
void print_debug(String message) {
#ifdef DEBUG_MODE
    Serial.println(message);
#endif
}
/*
void print_debug(const char* message) {
#ifdef DEBUG_MODE
    Serial.println(message);
#endif
}
*/




class Device {
//private: //*********************************************dopo fare i metodi getter e setter per ogni membro
public:
  String device_model;
  String device_id;
  String fixed_id;
  String location;
  float battery_level;
  String state;
  String ip_address;
  String audio_data_topic_key;
  String invitation_id;
  bool subscribed_flag;
  String communication_id;
  bool enable_audio_record;
  String last_record_date;
  unsigned int last_record_epoch;
  String net_password;
  //bool pending_subscription;
  //unsigned int last_subscription_attempt_epoch;
  bool start_recording_now;
  bool start_sending_audio_now;
  bool currently_sending_audio;
  bool new_audio_data_is_available;
  unsigned int deep_sleep_time;
  unsigned int max_time_on;
  unsigned int epoch_time_at_subscription;
  unsigned int max_time_for_successful_subscription;
  unsigned int record_time;
  unsigned int I2S_channel_num;
  unsigned int I2S_sample_rate;
  unsigned int I2S_sample_bits;
  unsigned int audio_record_size;
  unsigned int I2S_read_len;
  bool ready_to_set_device_flag;
  unsigned deep_sleep_current_time;
  bool deep_sleep_flag;

public:
  // Constructor with default values for fields
  // Constructor (same as before)
  Device() :
    device_model("ESP32_model_1"),
    device_id(""),
    fixed_id("f9-hfchxhjh"),
    location("Hive9"),
    battery_level(75.5),
    state("idle"),
    ip_address(""),
    audio_data_topic_key(""),
    invitation_id(""),
    subscribed_flag(false),
    communication_id(""),
    enable_audio_record(true),
    last_record_date(""),
    last_record_epoch(0),
    net_password(net_password),
    //pending_subscription(false),
    //last_subscription_attempt_epoch(0),
    start_recording_now(false),
    start_sending_audio_now(false),
    currently_sending_audio(false),
    new_audio_data_is_available(false),
    deep_sleep_time(60),
    max_time_on(180),
    epoch_time_at_subscription(0),
    max_time_for_successful_subscription(20),
    record_time(1),
    I2S_channel_num(1),
    I2S_sample_rate(22050),
    I2S_sample_bits(16),
    audio_record_size(I2S_channel_num * I2S_sample_rate * I2S_sample_bits / 8 * record_time),
    I2S_read_len(audio_record_size / 10),
    ready_to_set_device_flag(false),
    deep_sleep_current_time(570),
    deep_sleep_flag(false){}

    // GET METHODS  ///////////////////////////////////////////

    
    // UPDATE METHODS  ///////////////////////////////////////////


    void update_battery_level(float new_level) {
      print_debug("update_battery_level");
      if (new_level >= 0.0 && new_level <= 100.0) {
        this->battery_level = new_level;
      } else {
        Serial.println("Battery level must be between 0.0 and 100.0.");
      }
    }

    void update_audio_record_size() {
      print_debug("update_audio_record_size");
      // AUDIO_RECORD_SIZE (I2S_CHANNEL_NUM * I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 * RECORD_TIME)
      I2S_channel_num = this->I2S_channel_num;
      I2S_sample_rate = this->I2S_sample_rate;
      I2S_sample_bits = this->I2S_sample_bits;
      record_time = this->record_time;
      this->audio_record_size = I2S_channel_num * I2S_sample_rate * I2S_sample_bits / 8 * record_time;
    }

    void update_I2S_read_len() {
      print_debug("update_I2S_read_len");

      int lowerLimit = 1000;
      int upperLimit = 20000;                                                                             //esempio con audio_record_size = 44100
      int unsigned audio_record_size = (int)this->audio_record_size;                                      //44100
      int largestDivisor = -1; // Initialize to an invalid number

      for (int i = 1; i * i <= audio_record_size; i++) { // Loop from 1 to sqrt(n)                        // i=1            i=2             i=3  
          if (audio_record_size % i == 0) { // If i is a divisor                                          // 44100%1=0      44100%2=0       44100%3=0 
              // Check if it's in the specified range
              if (i >= lowerLimit && i <= upperLimit) {                                                   // fuori          fuori           fuori
                  largestDivisor = max(largestDivisor, i); // Update largest divisor                      // 1              2               3
              }
              
              // Check the complementary divisor
              int complementaryDivisor = audio_record_size / i;                                           // 44100/1=44100  44100/2=22050   44100/3=14700
              if (complementaryDivisor >= lowerLimit && complementaryDivisor <= upperLimit) {             // fuori          fuori           dentro
                  largestDivisor = max(largestDivisor, complementaryDivisor); // Update largest divisor   // 1              2               14700
              }
          }
      }
      this->I2S_read_len = largestDivisor;                                                                                                //14700
    }



    // DICTIONARY METHODS

    //String all_infos_to_dict() {
    DynamicJsonDocument all_infos_to_dict() {
      print_debug("all_infos_to_dict");
      // Generate a JSON string representation of the device info
      DynamicJsonDocument doc(512);

      doc["device_model"] = device_model;
      doc["device_id"] = device_id;
      doc["fixed_id"] = fixed_id;
      doc["location"] = location;
      doc["battery_level"] = battery_level;
      doc["state"] = state;
      doc["ip_address"] = ip_address;
      doc["enable_audio_record"] = enable_audio_record;
      doc["last_record_date"] = last_record_date;
      

      //String output;
      //serializeJson(doc, output); // Serialize JSON to a string
      //return output; // Return the JSON string
      return doc;
    }

};
