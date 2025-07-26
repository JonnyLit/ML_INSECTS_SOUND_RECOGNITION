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
    start_recording_now(false),
    start_sending_audio_now(false),
    currently_sending_audio(false),
    new_audio_data_is_available(false),
    deep_sleep_time(60),
    max_time_on(180),
    epoch_time_at_subscription(0),
    max_time_for_successful_subscription(20),
    record_time(10),
    I2S_channel_num(1),
    I2S_sample_rate(16000),
    I2S_sample_bits(16),
    audio_record_size(I2S_channel_num * I2S_sample_rate * I2S_sample_bits / 8 * record_time),
    I2S_read_len(audio_record_size / 10),
    ready_to_set_device_flag(false){}

    // GET METHODS  ///////////////////////////////////////////

    String get_device_model(){
      print_debug("get_device_model");
      return this->device_model;
    }

    String get_device_id(){
      print_debug("get_device_id");
      return this->device_id;
    }

    String get_fixed_id(){
      print_debug("get_fixed_id");
      return this->fixed_id;
    }

    String get_location(){
      print_debug("get_location");
      return this->location;
    }

    float get_battery_level(){
      print_debug("get_battery_level");
      return this->battery_level;
    }

    String get_state(){
      print_debug("get_state");
      return this->state;
    }

    String get_ip_address(){
      print_debug("get_ip_address");
      return this->ip_address;
    }

    String get_audio_data_topic_key(){
      print_debug("get_audio_data_topic_key");
      return this->audio_data_topic_key;
    }

    String get_invitation_id(){
      print_debug("get_invitation_id");
      return this->invitation_id;
    }

    bool get_subscribed_flag(){
      print_debug("get_subscribed_flag");
      return this->subscribed_flag;
    }

    String get_communication_id(){
      print_debug("get_communication_id");
      return this->communication_id;
    }

    bool get_enable_audio_record(){
      print_debug("get_enable_audio_record");
      return this->enable_audio_record;
    }

    String get_last_record_date(){
      print_debug("get_last_record_date");
      return this->last_record_date;
    }

    unsigned int get_last_record_epoch(){
      print_debug("get_last_record_epoch");
      return this->last_record_epoch;
    }

    String get_net_password(){
      print_debug("get_net_password");
      return this->net_password;
    }

    bool get_start_recording_now(){
      print_debug("get_start_recording_now");
      return this->start_recording_now;
    }

    bool get_start_sending_audio_now(){
      print_debug("get_start_sending_audio_now");
      return this->start_sending_audio_now;
    }

    bool get_currently_sending_audio(){
      print_debug("get_currently_sending_audio");
      return this->currently_sending_audio;
    }

    bool get_new_audio_data_is_available(){
      print_debug("get_new_audio_data_is_available");
      return this->new_audio_data_is_available;
    }

    unsigned int get_deep_sleep_time(){
      print_debug("get_deep_sleep_time");
      return this->deep_sleep_time;
    }

    unsigned int get_max_time_on(){
      print_debug("get_max_time_on");
      return this->max_time_on;
    }

    unsigned int get_epoch_time_at_subscription(){
      print_debug("get_epoch_time_at_subscription");
      return this->epoch_time_at_subscription;
    }

    unsigned int get_max_time_for_successful_subscription(){
      print_debug("get_max_time_for_successful_subscription");
      return this->max_time_for_successful_subscription;
    }

    unsigned int get_record_time(){
      print_debug("get_record_time");
      return this->record_time;
    }

    unsigned int get_I2S_channel_num(){
      print_debug("get_I2S_channel_num");
      return this->I2S_channel_num;
    }

    unsigned int get_I2S_sample_rate(){
      print_debug("get_I2S_sample_rate");
      return this->I2S_sample_rate;
    }

    unsigned int get_I2S_sample_bits(){
      print_debug("get_I2S_sample_bits");
      return this->I2S_sample_bits;
    }

    unsigned int get_audio_record_size(){
      print_debug("get_audio_record_size");
      return this->audio_record_size;
    }    

    unsigned int get_I2S_read_len(){
      print_debug("get_I2S_read_len");
      return this->I2S_read_len;
    }

    bool get_ready_to_set_device_flag(){
      print_debug("get_set_device_flag");
      return this->device_model;
    }
    
    // UPDATE METHODS  ///////////////////////////////////////////


    void update_device_model(const String& device_model) {
      print_debug("update_device_model");
      this->device_model = device_model;
    }

    void update_device_id(const String& device_id) {
      print_debug("update_device_id");
      this->device_id = device_id;
    }

    void update_fixed_id(const String& fixed_id) {
      print_debug("update_fixed_id");
      this->fixed_id = fixed_id;
    }

    void update_location(const String& location) {
      print_debug("update_location");
      this->location = location;
    }

    void update_battery_level(float new_level) {
      print_debug("update_battery_level");
      if (new_level >= 0.0 && new_level <= 100.0) {
        this->battery_level = new_level;
      } else {
        Serial.println("Battery level must be between 0.0 and 100.0.");
      }
    }

    void update_state(const String& state) {
      print_debug("update_state");
      this->state = state;
    }

    void update_ip_address(const String& ip_address) {
      print_debug("update_ip_address");
      this->ip_address = ip_address;
    }

    void update_audio_data_topic_key(const String& audio_data_topic_key) {
      print_debug("update_audio_data_topic_key");
      this->audio_data_topic_key = audio_data_topic_key;
    }

    void update_invitation_id(const String& invitation_id) {
      print_debug("update_invitation_id");
      this->invitation_id = invitation_id;
    }

    void update_subscribed_flag(bool subscribed_flag) {
      print_debug("update_subscribed_flag");
      this->subscribed_flag = subscribed_flag;
    }

    void update_communication_id(const String& communication_id) {
      print_debug("update_communication_id");
      this->communication_id = communication_id;
    }

    void update_enable_audio_record(bool enable_audio_record) {
      print_debug("update_enable_audio_record");
      this->enable_audio_record = enable_audio_record;
    }

    void update_last_record_date(const String& last_record_date) {
      print_debug("update_last_record_date");
      this->last_record_date = last_record_date;
    }

    void update_last_record_epoch(unsigned int last_record_epoch) {
      print_debug("update_last_record_epoch");
      this->last_record_epoch = last_record_epoch;
    }

    void update_net_password(const String& net_password) {
      print_debug("update_net_password");
      this->net_password = net_password;
    }

    void update_start_recording_now(bool start_recording_now) {
      print_debug("update_start_recording_now");
      this->start_recording_now = start_recording_now;
    }

    void update_start_sending_audio_now(bool start_sending_audio_now) {
      print_debug("update_start_sending_audio_now");
      this->start_sending_audio_now = start_sending_audio_now;
    }

    void update_currently_sending_audio(bool currently_sending_audio) {
      print_debug("update_currently_sending_audio");
      this->currently_sending_audio = currently_sending_audio;
    }

    void update_new_audio_data_is_available(bool new_audio_data_is_available) {
      print_debug("update_new_audio_data_is_available");
      this->new_audio_data_is_available = new_audio_data_is_available;
    }

    void update_deep_sleep_time(unsigned int deep_sleep_time) {
      print_debug("update_deep_sleep_time");
      this->deep_sleep_time = deep_sleep_time;
    }

    void update_max_time_on(unsigned int max_time_on) {
      print_debug("update_max_time_on");
      this->max_time_on = max_time_on;
    }

    void update_epoch_time_at_subscription(unsigned int epoch_time_at_subscription) {
      print_debug("update_epoch_time_at_subscription");
      this->epoch_time_at_subscription = epoch_time_at_subscription;
    }

    void update_max_time_for_successful_subscription(unsigned int max_time_for_successful_subscription) {
      print_debug("update_max_time_for_successful_subscription");
      this->max_time_for_successful_subscription = max_time_for_successful_subscription;
    }

    void update_record_time(unsigned int record_time) {
      print_debug("update_record_time");
      this->record_time = record_time;
    }

    void update_I2S_channel_num(unsigned int I2S_channel_num) {
      print_debug("update_I2S_channel_num");
      this->I2S_channel_num = I2S_channel_num;
    }

    void update_I2S_sample_rate(unsigned int I2S_sample_rate) {
      print_debug("update_I2S_sample_rate");
      this->I2S_sample_rate = I2S_sample_rate;
    }

    void update_I2S_sample_bits(unsigned int I2S_sample_bits) {
      print_debug("update_I2S_sample_bits");
      this->I2S_sample_bits = I2S_sample_bits;
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

    void update_ready_to_set_device_flag(bool ready_to_set_device_flag) {
      print_debug("update_ready_to_set_device_flag");
      this->ready_to_set_device_flag = ready_to_set_device_flag;
    }

    void update_audio_record_size(unsigned int audio_record_size){
      print_debug("update_audio_record_size");
      this->audio_record_size = audio_record_size;
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



    /*
    void update_communication_id(const String& communication_id);

    void update_invitation_id(const String& invitation_id);

    void update_audio_data_topic_key(const String& audio_data_topic_key);

    void update_device_id(const String& device_id);

    void update_enable_audio_record(bool enable_audio_record);

    void update_last_record_date(const String& last_record_date);

    void update_last_record_epoch(unsigned int last_record_epoch);

    void update_battery_level(float new_level);

    void update_state(const String& new_state);

    String all_infos_to_dict();

    String subscription_infos_to_dict();
    */


};
