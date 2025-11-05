import base64
import ssl
import subprocess
import time
import os
import random
import json
import sys
import paho.mqtt.client as mqtt
import logging
import credentials
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logging.basicConfig(level=logging.DEBUG)

DEBUG_MODE = True  # if False, there will be no prints


def print_debug(string_to_print):
    if DEBUG_MODE is True:
        print(string_to_print)


# Define the MQTT settings
broker = "test.mosquitto.org"  # Change as necessary
# broker = "broker.emqx.io"  # Change as necessary
# broker = "demo.thingsboard.io"  # Change as necessary
port = 1883  # Common port for MQTT or 8883 for TLS
keepalive = 40

# Specify the topics to subscribe to
topics = [
    ("NodeMCU_connect_to_sensors_net/ack", 2),  # <--
    ("NodeMCU_recording_problems", 2),  # <--
    ("audio/request", 2),  # <--
    ("audio/ack", 2),  # <--
    ("NodeMCU_get_infos", 2),  # <--
]


class Device:

    def __init__(self, device_model, device_id, location, battery_level, deep_sleep_time, fixed_id,
                 deep_sleep_simulation_time=42, deep_sleep_simulation_flag=False, record_time=1, reset_flag=False,
                 max_time_on=600, max_time_for_successful_subscription=10, epoch_time_at_subscription=None,
                 new_audio_data_is_available=False, currently_sending_audio=False, start_recording_now=False,
                 start_sending_audio_now=False, enable_audio_record=False, net_password="", subscribed_flag=False,
                 last_record_date=None, last_record_epoch=None, audio_data_topic_key=None, communication_id=None,
                 invitation_id=None, state=None, ip_address=None):
        """
        Initialize the device with relevant information.

        :param device_model: str - Model of the device
        :param device_id: str - Unique identifier for the device
        :param fixed_id:
        :param location: str - Physical location of the device
        :param battery_level: float - Current battery level percentage of the device (0.0 to 100.0)
        :param state: str - Current state of the device ("idle", "waiting_to_record_audio", "waiting_an_audio_request", "sending_audio", "sleeping", "recording_problems", "disconnecting", "resetting")
        :param ip_address
        :param audio_data_topic_key
        :param invitation_id
        :param subscribed_flag
        :param communication_id
        :param enable_audio_record
        :param last_record_date
        :param last_record_epoch
        :param net_password
        :param start_recording_now
        :param start_sending_audio_now
        :param currently_sending_audio
        :param new_audio_data_is_available
        :param deep_sleep_time
        :param max_time_on
        :param epoch_time_at_subscription
        :param max_time_for_successful_subscription
        :param reset_flag
        :param record_time
        :param deep_sleep_simulation_time
        :param deep_sleep_simulation_flag
        """

        self.device_model = device_model
        self.device_id = device_id
        self.fixed_id = fixed_id
        self.location = location
        self.battery_level = battery_level
        self.state = state
        self.ip_address = ip_address
        self.audio_data_topic_key = audio_data_topic_key
        self.invitation_id = invitation_id
        self.subscribed_flag = subscribed_flag
        self.communication_id = communication_id
        self.enable_audio_record = enable_audio_record
        self.last_record_date = last_record_date
        self.last_record_epoch = last_record_epoch
        self.net_password = net_password
        self.start_recording_now = start_recording_now
        self.start_sending_audio_now = start_sending_audio_now
        self.currently_sending_audio = currently_sending_audio
        self.new_audio_data_is_available = new_audio_data_is_available
        self.deep_sleep_time = deep_sleep_time
        self.max_time_on = max_time_on
        self.epoch_time_at_subscription = epoch_time_at_subscription
        self.max_time_for_successful_subscription = max_time_for_successful_subscription
        self.reset_flag = reset_flag
        self.record_time = record_time
        self.deep_sleep_simulation_time = deep_sleep_simulation_time
        self.deep_sleep_simulation_flag = deep_sleep_simulation_flag

    # GET METHODS

    def get_device_model(self):
        print("")
        return self.device_model

    def get_device_id(self):
        print("get_device_id")
        return self.device_id

    def get_fixed_id(self):
        print("get_fixed_id")
        return self.fixed_id

    def get_location(self):
        print("get_location")
        return self.location

    def get_battery_level(self):
        print("get_battery_level")
        return self.battery_level

    def get_state(self):
        print("get_state")
        return self.state

    def get_ip_address(self):
        print("get_ip_address")
        return self.ip_address

    def get_audio_data_topic_key(self):
        print("get_audio_data_topic_key")
        return self.audio_data_topic_key

    def get_invitation_id(self):
        print("get_invitation_id")
        return self.invitation_id

    def get_subscribed_flag(self):
        print("get_subscribed_flag")
        return self.subscribed_flag

    def get_communication_id(self):
        print("get_communication_id")
        return self.communication_id

    def get_enable_audio_record(self):
        print("get_enable_audio_record")
        return self.enable_audio_record

    def get_last_record_date(self):
        print("get_last_record_date")
        return self.last_record_date

    def get_last_record_epoch(self):
        print("get_last_record_epoch")
        return self.last_record_epoch

    def get_net_password(self):
        print("get_net_password")
        return self.net_password

    def get_start_recording_now(self):
        print("get_start_recording_now")
        return self.start_recording_now

    def get_start_sending_audio_now(self):
        print("get_start_sending_audio_now")
        return self.start_sending_audio_now

    def get_currently_sending_audio(self):
        print("get_currently_sending_audio")
        return self.currently_sending_audio

    def get_new_audio_data_is_available(self):
        print("get_new_audio_data_is_available")
        return self.new_audio_data_is_available

    def get_deep_sleep_time(self):
        print("get_deep_sleep_time")
        return self.deep_sleep_time

    def get_max_time_on(self):
        print("get_max_time_on")
        return self.max_time_on

    def get_epoch_time_at_subscription(self):
        print("get_epoch_time_at_subscription")
        return self.epoch_time_at_subscription

    def get_max_time_for_successful_subscription(self):
        print("get_max_time_for_successful_subscription")
        return self.max_time_for_successful_subscription

    def get_record_time(self):
        print("get_record_time")
        return self.record_time

    # UPDATE METHODS

    def update_device_model(self, device_model):
        print_debug("update_device_model")
        self.device_model = device_model

    def update_device_id(self, device_id):
        print_debug("update_device_id")
        self.device_id = device_id

    def update_fixed_id(self, fixed_id):
        print_debug("update_fixed_id")
        self.fixed_id = fixed_id

    def update_location(self, location):
        print_debug("update_location")
        self.location = location

    def update_battery_level(self, battery_level):
        print_debug("update_battery_level")
        self.battery_level = battery_level
        """
        if 0.0 <= battery_level <= 100.0:
            self.battery_level = battery_level
        else:
            raise ValueError("Battery level must be between 0.0 and 100.0.")
        """

    def update_state(self, state):
        print_debug("update_state")
        self.state = state

    def update_ip_address(self, ip_address):
        print_debug("update_ip_address")
        self.ip_address = ip_address

    def update_audio_data_topic_key(self, audio_data_topic_key):
        print_debug("update_audio_data_topic_key")
        self.audio_data_topic_key = audio_data_topic_key

    def update_invitation_id(self, invitation_id):
        print_debug("update_invitation_id")
        self.invitation_id = invitation_id

    def update_subscribed_flag(self, subscribed_flag):
        print_debug("update_subscribed_flag")
        self.subscribed_flag = subscribed_flag

    def update_communication_id(self, communication_id):
        print_debug("update_communication_id")
        self.communication_id = communication_id

    def update_enable_audio_record(self, enable_audio_record):
        print_debug("update_enable_audio_record")
        self.enable_audio_record = enable_audio_record

    def update_last_record_date(self, last_record_date):
        print_debug("update_last_record_date")
        self.last_record_date = last_record_date

    def update_last_record_epoch(self, last_record_epoch):
        print_debug("update_last_record_epoch")
        self.last_record_epoch = last_record_epoch

    def update_net_password(self, net_password):
        print_debug("update_net_password")
        self.net_password = net_password

    def update_start_recording_now(self, start_recording_now):
        print_debug("update_start_recording_now")
        self.start_recording_now = start_recording_now

    def update_start_sending_audio_now(self, start_sending_audio_now):
        print_debug("update_start_sending_audio_now")
        self.start_sending_audio_now = start_sending_audio_now

    def update_currently_sending_audio(self, currently_sending_audio):
        print_debug("update_currently_sending_audio")
        self.currently_sending_audio = currently_sending_audio

    def update_new_audio_data_is_available(self, new_audio_data_is_available):
        print_debug("update_new_audio_data_is_available")
        self.new_audio_data_is_available = new_audio_data_is_available

    def update_deep_sleep_time(self, deep_sleep_time):
        print_debug("update_deep_sleep_time")
        self.deep_sleep_time = deep_sleep_time

    def update_max_time_on(self, max_time_on):
        print_debug("update_max_time_on")
        self.max_time_on = max_time_on

    def update_epoch_time_at_subscription(self, epoch_time_at_subscription):
        print_debug("update_epoch_time_at_subscription")
        self.epoch_time_at_subscription = epoch_time_at_subscription

    def update_max_time_for_successful_subscription(self, max_time_for_successful_subscription):
        print_debug("update_max_time_for_successful_subscription")
        self.max_time_for_successful_subscription = max_time_for_successful_subscription

    def update_record_time(self, record_time):
        print_debug("update_record_time")
        self.record_time = record_time

    # DICTIONARY METHODS

    def all_infos_to_dict(self):
        print_debug("all_infos_to_dict")
        """
        Convert the Client object fields into a JSON string.

        :return: str - JSON string representation of the Client object
        """

        battery_level = random.choice([2.9, 3, 3.1, 3.3, 3.5, 3.7, 3.9, 4.2])
        esp32_device.update_battery_level(battery_level)

        esp32_data = {
            "device_model": self.device_model,
            "device_id": self.device_id,
            "fixed_id": self.fixed_id,
            "location": self.location,
            "battery_level": self.battery_level,
            "state": self.state,
            "ip_address": self.ip_address,
            "enable_audio_record": self.enable_audio_record,
            "last_record_date": self.last_record_date
        }
        return esp32_data  # Return the dictionary directly instead of JSON string

    def subscription_infos_to_dict(self):
        print_debug("subscription_infos_to_dict")
        """
        Convert the Client object fields into a JSON string.

        :return: str - JSON string representation of the Client object
        """
        esp32_data = {
            "device_model": self.device_model,
            "location": self.location,
            "battery_level": self.battery_level,
            "state": self.state,
            "ip_address": self.ip_address,
            "enable_audio_record": self.enable_audio_record,
            "last_record_date": self.last_record_date
        }
        return esp32_data  # Return the dictionary directly instead of JSON string

    def __str__(self):
        return f"Client(device_model: {self.device_model}, device_id: {self.device_id}, fixed_id: {self.fixed_id}, location: {self.location}, battery_level: {self.battery_level}%, state: {self.state}, ip_address: {self.ip_address}, audio_data_topic_key: {self.audio_data_topic_key}, invitation_id: {self.invitation_id}, subscribed_flag: {self.subscribed_flag}, communication_id: {self.communication_id}, enable_audio_record: {self.enable_audio_record}, last_record_date: {self.last_record_date}, last_record_epoch: {self.last_record_epoch}, net_password: {self.net_password}, start_recording_now: {self.start_recording_now}, start_sending_audio_now: {self.start_sending_audio_now}, currently_sending_audio: {self.currently_sending_audio}, new_audio_data_is_available: {self.new_audio_data_is_available}, deep_sleep_time: {self.deep_sleep_time}, max_time_on: {self.max_time_on}, epoch_time_at_subscription: {self.epoch_time_at_subscription}, max_time_for_successful_subscription: {self.max_time_for_successful_subscription})"


# Generate a random integer ID between 1000 and 9999
random_id = random.randint(1000, 9999)
print_debug(random_id)

import random
import string


def generate_random_code(length):
    print_debug("generate_random_code")
    # Create a pool of uppercase, lowercase letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random code by selecting random characters from the pool
    random_code = ''.join(random.choice(characters) for _ in range(length))
    return random_code


# Generate a 20-digit random alphanumeric code
# code = generate_random_code(20)
# print_debug("Random Alphanumeric Code:", code)


def reset_device():
    print_debug("Resetting device")  # TODO


def publish_infos_with_communication_id(client, topic, Device, communication_id):
    print_debug("publish_infos_with_communication_id")
    infos_content = Device.all_infos_to_dict()
    message = {"infos": infos_content, "communication_id": communication_id, "simulation": True}
    message = json.dumps(message)
    Device.update_communication_id(communication_id)
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug("+++++++++++++++++++++++++++++")
    print_debug(f"formatted_time: {formatted_time}")
    print_debug(f"topic: {topic}")
    print_debug(f"Sending: {message}")
    encrypted_message = encrypt(message)
    client.publish(topic, encrypted_message, qos=2)  # client


def publish_infos_with_invitation_id(client, topic, Device, invitation_id):
    print_debug("publish_infos_with_invitation_id")
    # infos_content = Device.subscription_infos_to_dict
    infos_content = Device.all_infos_to_dict()
    message = {"infos": infos_content, "invitation_id": invitation_id, "net_password": Device.get_net_password(),
               "simulation": True}
    message = json.dumps(message)
    Device.update_invitation_id(invitation_id)
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug("+++++++++++++++++++++++++++++")
    print_debug(f"formatted_time: {formatted_time}")
    print_debug(f"topic: {topic}")
    print_debug(f"Sending: {message}")
    encrypted_message = encrypt(message)
    client.publish(topic, encrypted_message, qos=2)  # client


def publish_wav_file(client, topic, file_path):
    print_debug("publish_wav_file")
    # client_id = "hsd72'f934jna°,f#3qqvjHM"  # all clients must have a different client_id ******************************
    # Create an MQTT client instance
    # client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5) #******************
    # client = mqtt.Client(protocol=mqtt.MQTTv5)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)  #*****************
    try:  # ***************************

        # client.loop_start() #*****************************
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
            # encoded_string = base64.b64encode(audio_data).decode('utf-8') #*************
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            microseconds = int((time.time() % 1) * 1_000_000)
            # Combine formatted time with microseconds
            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
            print_debug("+++++++++++++++++++++++++++++")
            print_debug(f"formatted_time when published: {formatted_time_with_micros}")
            print_debug(f"topic: {topic}")
            print_debug(f"Published {os.path.basename(file_path)} to topic '{topic}'")
            # client.connect("broker.emqx.io", 1883) #*****************************
            # client.loop_start()  # *****************************
            # client.subscribe(topic, qos=2)
            client.publish(topic, audio_data, qos=2)  # client
            # client.publish(topic, encoded_string)
            logging.debug(f"Published audio data of size: {len(audio_data)}bytes")  # ********************
    except Exception as e:  # *****************************
        logging.error(f"Error: {e}")  # *****************************


def notify_available_audio_with_communication_id(client, topic, Device, communication_id):
    print_debug("notify_available_audio_with_communication_id")
    publish_infos_with_communication_id(client, topic, Device, communication_id)


def record_audio():
    print_debug("record_audio")
    '''
    valid_audio = True # to be changed in test phase: flag True/False to assert the audio is valid/invalid
    max_recording_attempts = 3
    iter_attempts = 1
    while iter_attempts <= max_recording_attempts and valid_audio is False:
        # Start recording
        # Audio validation (check audio correctness) --> if validation is successfully (True) then continue, otherwise (False) retry other 3 times and in the worst case notify the problem
        ###############################################################################
        #The audio recorded can be correct as expected (True) or not (False)
        valid_audio = False # change it to the preferred value for the simulation test (valid_audio = False to simulate a recording problem)
        print_debug("valid_audio: ", valid_audio)
        ###############################################################################
        iter_attempts += 1

    communication_id = generate_random_code(20)
    if valid_audio is False:
        notify_available_audio_with_communication_id("Raspberry_recording_problems", Device, communication_id)
    else:
        notify_available_audio_with_communication_id("Raspberry_notify_audio_available", Device, communication_id)
    '''


def encrypt(message):
    # Generate a random IV
    iv = os.urandom(16)

    # Pad the message with null bytes to make its length a multiple of the block size
    padded_data = message.encode() + b'\x00' * (
            (algorithms.AES.block_size // 8) - len(message) % (algorithms.AES.block_size // 8))

    # Create AES cipher in CBC mode
    cipher = Cipher(algorithms.AES(credentials.aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()

    # Return IV + encrypted data as Base64
    return base64.b64encode(iv + encrypted).decode('utf-8')


def decrypt(iv, encrypted_data):
    # Create AES cipher in CBC mode for decryption
    cipher = Cipher(algorithms.AES(credentials.aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove null padding
    unpadded_data = decrypted_padded.rstrip(b'\x00')  # Strip away null bytes from the end

    return unpadded_data.decode('utf-8')  # Decode to string


def is_valid_json(string):
    try:
        json.loads(string)  # Try to parse the string as JSON
        return True  # Return True if successful
    except json.JSONDecodeError:
        return False  # Return False if there is a parsing error


esp32_device = Device(
    device_model="ESP32_model_1",
    device_id=None,
    fixed_id="f3-dsfagafvbcv",
    location="Hive3",
    battery_level=3.3,
    state="idle",
    ip_address="192.168.1.100",
    audio_data_topic_key=None,
    invitation_id=None,
    subscribed_flag=False,
    communication_id=None,
    enable_audio_record=True,
    last_record_date=None,
    last_record_epoch=None,
    net_password="sensors_network_password",
    start_recording_now=False,
    start_sending_audio_now=False,
    currently_sending_audio=False,
    new_audio_data_is_available=False,
    deep_sleep_time=120,
    max_time_on=260,
    epoch_time_at_subscription=0,
    max_time_for_successful_subscription=20,
    reset_flag=False,
    record_time=1,
    deep_sleep_simulation_flag=False,
    deep_sleep_simulation_time=42
)


def set_device_default_values(Device):
    Device.device_model = "ESP32_model_1"
    Device.device_id = None
    Device.fixed_id = "f3-dsfagafvbcv"
    Device.location = "Hive3"
    Device.battery_level = 3.3
    Device.state = "idle"
    Device.ip_address = "192.168.1.100"
    Device.audio_data_topic_key = None
    Device.invitation_id = None
    Device.subscribed_flag = False
    Device.communication_id = None
    Device.enable_audio_record = True
    Device.last_record_date = None
    Device.last_record_epoch = None
    Device.net_password = "sensors_network_password"
    Device.start_recording_now = False
    Device.start_sending_audio_now = False
    Device.currently_sending_audio = False
    Device.new_audio_data_is_available = False
    Device.deep_sleep_time = 120
    Device.max_time_on = 180
    Device.epoch_time_at_subscription = 0
    Device.max_time_for_successful_subscription = 20
    Device.reset_flag = False
    Device.record_time = 1
    Device.deep_sleep_simulation_flag = False,
    Device.deep_sleep_simulation_time = 42


def simulate_deep_sleep(client, sleep_time_seconds):
    print("simulate_deep_sleep(client, sleep_time_seconds)")
    # Unsubscribing from all the topics, to properly simulate a deep sleep mode

    for topic, qos in topics:
        client.unsubscribe(topic)
        print_debug(f"Unsubscribed from topic: {topic}")

    print(f"!!!!!!!!!!!!!!!!!  Going to sleep for: {int(sleep_time_seconds)} seconds!!!!!!!!!!!!!!!!!!!!")
    # Now go to deep sleep
    for i in range(1, int(sleep_time_seconds) + 1):
        time.sleep(1)
        print(f"{i} seconds_____________________________________________sleeping...")

    # Resetting the main function
    esp32_device.reset_flag = True


'''
def simulate_deep_sleep(client, sleep_time_seconds):
    print("simulate_deep_sleep(client, sleep_time_seconds)")
    # Unsubscribing from all the topics, to properly simulate a deep sleep mode

    for topic, qos in topics:
        client.unsubscribe(topic)
        print_debug(f"Unsubscribed from topic: {topic}")

    print(f"!!!!!!!!!!!!!!!!!  Going to sleep for: {int(sleep_time_seconds)} seconds!!!!!!!!!!!!!!!!!!!!")
    # Now go to deep sleep
    for i in range(1, int(sleep_time_seconds)+1):
        time.sleep(1)
        print(f"{i} seconds_____________________________________________sleeping...")

    # Resetting the main function
    esp32_device.reset_flag = True
'''


def on_log(client, userdata, level, buf):
    print_debug(f"log: {buf}")


'''
def on_disconnect(client, userdata, rc, properties=None):
    print_debug(f"Disconnected with result code {rc}")
    if rc != 0:
        print_debug("Unexpected disconnection. Trying to reconnect...")
        client.reconnect()
'''


# Callback function when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc, properties=None):
    print_debug("on_connect")
    # print_debug(f"Connected to broker at {broker} with result code {rc}")
    logging.debug(f"Connected to broker at {broker} with result code {rc}")
    if rc == 0:
        print_debug("Connection successful!")
        for topic, qos in topics:
            client.subscribe(topic, qos)
            print_debug(f"Subscribed to topic: {topic} with qos={qos}")
    elif rc == 1:
        print_debug("Connection refused – unacceptable protocol version.")
    elif rc == 2:
        print_debug("Connection refused – identifier rejected.")
    elif rc == 3:
        print_debug("Connection refused – broker unavailable.")
    elif rc == 4:
        print_debug("Connection refused – bad user name or password.")
    elif rc == 5:
        print_debug("Connection refused – not authorized.")
    else:
        print_debug(f"Unknown error code: {rc}.")


def clean_json_string(s: str) -> str:
    # Find the last occurrence of '}'
    last_brace_index = s.rfind('}')
    # If '}' is found, truncate the string
    if last_brace_index != -1:
        return s[:last_brace_index + 1]
    # If no '}' is found, return the original string or handle it as needed
    return s


# Callback function when a message is received.
def on_message(client, userdata, msg):
    # def on_message(client, userdata, msg, esp32_device):
    print_debug("on_message")
    # print_debug(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")

    if msg.topic == "NodeMCU_connect_to_sensors_net/ack":
        print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_debug("if msg.topic == NodeMCU_connect_to_sensors_net/ack:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 2 {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>, recording_frequency<freq>}<----------------------------------------------

        encoded_message = msg.payload.decode()
        dummy_json = '{"dummy":""}'
        json_received = json.loads(dummy_json)
        # DECRYPTION #####################################################################
        # Decode the Base64 message
        try:
            raw_data = base64.b64decode(encoded_message)

            iv = raw_data[:16]  # Extract the IV
            encrypted_data = raw_data[16:]  # Extract the encrypted data

            # Decrypt the message
            decrypted_message = decrypt(iv, encrypted_data)

            print("Raw Decrypted Message:", decrypted_message)
            print(f"Length of Raw Decrypted Message: {len(decrypted_message)}")

            if is_valid_json(decrypted_message):
                print("Valid Decrypted Message JSON:", decrypted_message)
                json_received = json.loads(decrypted_message)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")
        # END OF DECRYPTION ##############################################################

        try:
            if json_received["invitation_id"] == esp32_device.get_invitation_id():
                print_debug("if json_received[invitation_id] == esp32_device.invitation_id:")
                if json_received["Connection_permission"] == "allowed" or json_received[
                    "Connection_permission"] == "already_in":
                    print_debug(
                        "if json_received[Connection_permission] == allowed or json_received[Connection_permission] == already_in:")
                    # if json_received["Connection_permission"] == "allowed" or json_received["Connection_permission"] == "already_in":
                    esp32_device.update_device_id(
                        json_received["device_id"])  # Assignment of the Device ID to the device in the sensors network
                    esp32_device.update_subscribed_flag(True)

                    # if json_received["Connection_permission"] == "allowed":

                    print_debug(
                        "if json_received['Connection_permission'] == 'allowed' or json_received['Connection_permission'] == 'already_in':")
                    # print_debug("if json_received['Connection_permission'] == 'allowed':")
                    current_raspberrypi_timer = json_received["current_timer"]
                    deep_sleep_time = json_received["deep_sleep_time"]
                    print_debug(
                        f"current_raspberrypi_timer: {current_raspberrypi_timer} and deep_sleep_time: {deep_sleep_time}")

                    ''' EXPLANATION FOR THE NEXT LINES OF CODE: 
                    More or less, the deep_sleep_time given by the Raspberrypi to the esp32 is about 30sec less 
                    than the time between two consecutive measures (time between two consecutive audio requests)...
                    -So, if the esp32 tries to subscribe to the sensors network, and for example the audio request has to be done in 10 minutes by the Raspberrypi,
                    then the esp32 should stay on waiting for the audio request for more than 9 min... 
                    -To avoid this problem, whenever the esp32 tries to subscribe, the Raspberrypi gives to it the information about its current_timer for the 
                    audio request (the timer goes from 0 to time_between_two_consecutive_audio_requests),
                    so that the esp32 can go to sleep a fair amount of time before the audio request, instead of staying on all that time, wasting energy.
                    -Moreover, since from the Thinsgsboard dashboard it is possible to deactivate/activate any esp32 in the net, in order to not wait too much time
                    for the devices to go from OFF to ON, or ON to OFF, the max amount of time that the esp32 could sleep is 120 seconds.
                    -So, for example, if time_between_two_consecutive_audio_requests = 600 (10 minutes), then if the esp32 initially tries to subscribe when the 
                    current_raspberrypi_timer = 10 sec, the esp32 will go to sleep for 120 sec, then retry to subscribe (at that time the current_raspberrypi_timer would be about 130/140sec),
                    then it will go to sleep again for 120 sec, ...and so on till at the subscription attempt it will result a current_raspberrypi_timer near 600 sec, and the esp32 finally can
                    stay on, record the audio, send a notification to the Raspberrypi for an available audio, and then send the audio when requested, then going to sleep again for 120 sec, 
                    and so on with the same procedure...In this way, the esp32 doesn't waste energy staying on a lot of time before the audio request, and at the same time
                    from the Thingsboard Dashboard, if we try to toggle an esp32 (ON-->OFF or OFF-->ON), the device toggles in less than 120 seconds.

                    deep_sleep_time(DST)    current_raspberrypi_timer(CRT)      time_to_sleep(TTS)
                    60                      10                                  DST - CRT = 50 --> (<=120) -->N.A.
                    100                     10                                  DST - CRT = 90 --> (<=120) -->N.A.
                    120                     10                                  DST - CRT = 110 --> (<=120) -->N.A.
                    200                     10                                  DST - CRT = 190 --> (>120) --> TTS = DST - CRT - 120 = 70 --> (TTS <= 120) --> TTS = 70
                    240                     10                                  DST - CRT = 230 --> (>120) --> TTS = DST - CRT - 120 = 110 --> (TTS <= 120) --> TTS = 110
                    500                     10                                  DST - CRT = 490 --> (>120) --> TTS = DST - CRT - 120 = 370 --> (TTS > 120) --> TTS = 120
                    1000                    10                                  DST - CRT = 990 --> (>120) --> TTS = DST - CRT - 120 = 870 --> (TTS > 120) --> TTS = 120
                    '''

                    seconds_before_audio_request = 1  # it was 120 originally
                    if deep_sleep_time - current_raspberrypi_timer > seconds_before_audio_request:
                        print("if deep_sleep_time - current_raspberrypi_timer > seconds_before_audio_request:")
                        time_to_sleep = deep_sleep_time - current_raspberrypi_timer - seconds_before_audio_request

                        publish_infos_with_invitation_id(client, "Raspberry_connect_to_sensors_net/ack", esp32_device,
                                                         json_received[
                                                             "invitation_id"])  # the Raspberry Py will update the infos of this accepted new device
                        '''
                        if time_to_sleep > 120:
                            print("if time_to_sleep > 120:")
                            time_to_sleep = 120
                        if json_received["Connection_permission"] == "allowed":
                            print("if json_received['Connection_permission'] == 'allowed':")
                            publish_infos_with_invitation_id(client, "Raspberry_connect_to_sensors_net/ack",esp32_device, json_received["invitation_id"])  # the Raspberry Py will update the infos of this accepted new device
                        '''

                        # time.sleep(1) # this is important since in simulate_deep_sleep we are going tu unsubscribe, and the mqtt message previously sent could not reach the raspberrypi in time...
                        esp32_device.deep_sleep_simulation_time = time_to_sleep
                        esp32_device.deep_sleep_simulation_flag = True
                        # simulate_deep_sleep(client, time_to_sleep)
                        # esp32_device.reset_flag = True
                        print("client.loop_stop()")
                        # client.loop_stop()  # Stop the loop
                        print("client.disconnect()")
                        # client.disconnect()  # Disconnect from the broker
                        # return False

                    else:
                        print_debug("elif deep_sleep_time - current_raspberrypi_timer <= seconds_before_audio_request:")
                        # esp32_device.update_subscribed_flag(True)
                        # esp32_device.update_device_group(json_received["device_group"]) #Assigning a group between 1/2/3/4, which the device from now on will belong to (the group can change for next device_group assignments done through the "set_device_group" topic)
                        publish_infos_with_invitation_id(client, "Raspberry_connect_to_sensors_net/ack", esp32_device,
                                                         json_received[
                                                             "invitation_id"])  # the Raspberry Py will update the infos of this accepted new device
                        esp32_device.update_start_recording_now(True)
                        esp32_device.update_deep_sleep_time(json_received["deep_sleep_time"])
                        esp32_device.update_max_time_on(json_received["max_time_on"])
                        esp32_device.update_record_time(json_received["record_time"])
                elif json_received["Connection_permission"] == "denied":
                    esp32_device.update_subscribed_flag(False)
                    esp32_device.update_deep_sleep_time(json_received["deep_sleep_time"])
                    esp32_device.update_max_time_on(json_received["max_time_on"])
                    # simulate_deep_sleep(client, esp32_device.get_deep_sleep_time())
                    # simulate_deep_sleep(client, 120)
                    esp32_device.deep_sleep_simulation_time = esp32_device.get_deep_sleep_time()
                    esp32_device.deep_sleep_simulation_flag = True
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("ESP32: problem receiving the package for the topic 'NodeMCU_connect_to_sensors_net/ack'")
        print_debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")



    elif msg.topic == "NodeMCU_recording_problems":
        print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_debug("elif msg.topic == NodeMCU_recording_problems:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1      ------------------------------------>{communication_id:<ID>,infos:{infos}}
        # 2      {communication_id:<ID>}<--------------------------------------------------
        # 3      after this, the device will go in state "RECORDING_PROBLEMS", where the only enabled topics will be sleep_mode/reset/disconnection_from_broker/get_state
        # 4      after that, the RaspberryPy could optionally use one topic between: sleep_mode/reset/disconnection_from_broker/get_state

        encoded_message = msg.payload.decode()
        dummy_json = '{"dummy":""}'
        json_received = json.loads(dummy_json)
        # DECRYPTION #####################################################################
        # Decode the Base64 message
        try:
            raw_data = base64.b64decode(encoded_message)

            iv = raw_data[:16]  # Extract the IV
            encrypted_data = raw_data[16:]  # Extract the encrypted data

            # Decrypt the message
            decrypted_message = decrypt(iv, encrypted_data)

            print("Raw Decrypted Message:", decrypted_message)
            print(f"Length of Raw Decrypted Message: {len(decrypted_message)}")

            if is_valid_json(decrypted_message):
                print("Valid Decrypted Message JSON:", decrypted_message)
                json_received = json.loads(decrypted_message)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            if json_received["communication_id"] == esp32_device.get_communication_id():
                esp32_device.update_enable_audio_record(False)
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("ESP32: problem receiving the package for the topic 'NodeMCU_recording_problems'")
        print_debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    elif msg.topic == "audio/request":
        print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_debug("elif msg.topic == audio/request:")
        print_debug(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1 {device_id:<ID>, dynamic_topic_key:<topic_key>, communication_id:<ID>}<--------------------------------------------------

        encoded_message = msg.payload.decode()
        dummy_json = '{"dummy":""}'
        json_received = json.loads(dummy_json)
        # DECRYPTION #####################################################################
        # Decode the Base64 message
        try:
            raw_data = base64.b64decode(encoded_message)

            iv = raw_data[:16]  # Extract the IV
            encrypted_data = raw_data[16:]  # Extract the encrypted data

            # Decrypt the message
            decrypted_message = decrypt(iv, encrypted_data)

            print("Raw Decrypted Message:", decrypted_message)
            print(f"Length of Raw Decrypted Message: {len(decrypted_message)}")

            if is_valid_json(decrypted_message):
                print("Valid Decrypted Message JSON:", decrypted_message)
                json_received = json.loads(decrypted_message)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            if json_received["device_id"] == esp32_device.get_device_id():
                esp32_device.update_communication_id(json_received["communication_id"])
                esp32_device.update_audio_data_topic_key(json_received["audio_data_topic_key"])
                esp32_device.update_start_sending_audio_now(True)
                # 2      -------------------------------------------------------------------------------------------->{audio_content}

        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("ESP32: problem receiving the package for the topic 'audio/request'")
        print_debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    elif msg.topic == "audio/ack":
        print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_debug("elif msg.topic == audio/ack:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1  {communication_id:<ID>, dynamic_topic_key:<topic_key>, ack:ok/resend}<-----------------------------------

        encoded_message = msg.payload.decode()
        dummy_json = '{"dummy":""}'
        json_received = json.loads(dummy_json)
        # DECRYPTION #####################################################################
        # Decode the Base64 message
        try:
            raw_data = base64.b64decode(encoded_message)

            iv = raw_data[:16]  # Extract the IV
            encrypted_data = raw_data[16:]  # Extract the encrypted data

            # Decrypt the message
            decrypted_message = decrypt(iv, encrypted_data)

            print("Raw Decrypted Message:", decrypted_message)
            print(f"Length of Raw Decrypted Message: {len(decrypted_message)}")

            if is_valid_json(decrypted_message):
                print("Valid Decrypted Message JSON:", decrypted_message)
                json_received = json.loads(decrypted_message)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            if json_received["communication_id"] == esp32_device.get_communication_id():
                if json_received["ack"] == "ok":
                    print_debug("if json_received[ack] == ok:")
                    # simulate_deep_sleep(client, esp32_device.get_deep_sleep_time())
                    # simulate_deep_sleep(client, 120)
                    esp32_device.deep_sleep_simulation_time = esp32_device.get_deep_sleep_time()
                    esp32_device.deep_sleep_simulation_flag = True
                elif json_received["ack"] == "resend":
                    print_debug("elif json_received[ack] == resend:")
                    esp32_device.update_communication_id(json_received["communication_id"])
                    esp32_device.update_audio_data_topic_key(json_received["audio_data_topic_key"])
                    esp32_device.update_start_sending_audio_now(True)
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("ESP32: problem receiving the package for the topic 'audio/ack'")
        print_debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    # elif msg.topic == "audio/payload" + topic_key:                # only for the RaspberryPy

    elif msg.topic == "NodeMCU_get_infos":
        print_debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print_debug("elif msg.topic == NodeMCU_get_infos:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------

        encoded_message = msg.payload.decode()
        dummy_json = '{"dummy":""}'
        json_received = json.loads(dummy_json)
        # DECRYPTION #####################################################################
        # Decode the Base64 message
        try:
            raw_data = base64.b64decode(encoded_message)

            iv = raw_data[:16]  # Extract the IV
            encrypted_data = raw_data[16:]  # Extract the encrypted data

            # Decrypt the message
            decrypted_message = decrypt(iv, encrypted_data)

            print("Raw Decrypted Message:", decrypted_message)
            print(f"Length of Raw Decrypted Message: {len(decrypted_message)}")

            if is_valid_json(decrypted_message):
                print("Valid Decrypted Message JSON:", decrypted_message)
                json_received = json.loads(decrypted_message)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            if json_received["device_id"] == esp32_device.get_device_id():
                esp32_device.update_communication_id(json_received["communication_id"])
                publish_infos_with_communication_id(client, "Raspberry_get_infos", esp32_device,
                                                    json_received["communication_id"])
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("ESP32: problem receiving the package for the topic 'NodeMCU_get_infos'")
        print_debug("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    print("EXIT FROM on_message__________________________________________________________________________")


def simulate_publish_audio(client, Device):
    print("simulate_publish_audio")
    file_path = "test_audio.wav"  # Change to your .wav file path
    dynamic_topic = "audio/payload" + Device.get_audio_data_topic_key()
    publish_wav_file(client, dynamic_topic, file_path)


def simulate_record_and_notify(client, recording_seconds, Device):
    print("simulate_record_and_notify")
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug(f"formatted_time: {formatted_time}")
    print(f"Recording started (it will end in {recording_seconds} seconds)")
    for i in range(1, Device.get_record_time()):
        time.sleep(1)
        print(f"{i} seconds")
    print(f"Recording finished!")
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug(f"formatted_time: {formatted_time}")
    esp32_device.update_new_audio_data_is_available(True)
    esp32_device.update_last_record_epoch(time.time())
    esp32_device.update_last_record_date(formatted_time)
    time.sleep(1)  # just to be sure the raspberry received the previous "Raspberry_connect_to_sensors_net/ack"
    publish_infos_with_communication_id(client, "Raspberry_audio_available", esp32_device, generate_random_code(20))
    time.sleep(2)  # just to be sure the raspberry receives this message
    print("Exit from i2s_record_and_notify")


def main():
    mqtt_client_id = generate_random_code(12)  # all clients must have a different client_id

    # Create an MQTT client instance
    # client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
    client = mqtt.Client(client_id=mqtt_client_id, protocol=mqtt.MQTTv5)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    # Set the username and password
    # client.username_pw_set("your_username", "your_password")

    # Set SSL parameters
    '''
    client.tls_set(ca_certs="path/to/ca.crt",  # CA file
                   certfile="path/to/client.crt",  # Client certificate
                   keyfile="path/to/client.key",  # Client private key
                   tls_version=ssl.PROTOCOL_TLS)
    '''

    # Attach the callback functions
    print_debug("# Attach the callback functions")
    client.on_log = on_log
    client.on_connect = on_connect
    # client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Connect to the broker
    try:
        client.connect(broker, port, keepalive)
        # client.connect("broker.emqx.io", 1883, 60)
        print_debug(f"Successfully connected to the broker {broker}")
        # Start the loop in a separate thread to process network traffic and dispatch callbacks
        print_debug("client.loop_start()")
        client.loop_start()
    except Exception as e:
        print_debug(f"Could not connect to broker: {e}")
        logging.error(f"Exception occurred: {e}")
        exit()

    print("[SETUP] WAIT 3 sec")
    time.sleep(3)

    # Start asking to be part of the sensors network
    # subscribed_flag=False

    publish_infos_with_invitation_id(client, "Raspberry_connect_to_sensors_net/request", esp32_device,
                                     generate_random_code(20))
    esp32_device.update_epoch_time_at_subscription(time.time())  # epoch time at the first subscription attempt
    print(f"esp32_device.invitation_id: {esp32_device.get_invitation_id()}")
    print_debug("EXIT FROM void setup()")

    time_last_loop = 0
    deep_sleep_simulation_flag = False

    try:
        while True:

            if esp32_device.deep_sleep_simulation_flag is True:
                esp32_device.deep_sleep_simulation_flag = False
                simulate_deep_sleep(client, esp32_device.deep_sleep_simulation_time)

            if esp32_device.reset_flag is True:
                print("if reset_flag is True:")
                print("client.loop_stop()")
                print("client.disconnect()")
                client.loop_stop()  # Stop the loop
                client.disconnect()  # Disconnect from the broker
                esp32_device.reset_flag = False
                return False

            if time.time() - time_last_loop >= 1:
                print_debug("if time.time() - time_last_loop >= 1")
                time_last_loop = time.time()

                if esp32_device.get_subscribed_flag() is True:
                    print("if esp32_device.subscribed_flag is True")
                    if time.time() - esp32_device.get_epoch_time_at_subscription() >= esp32_device.get_max_time_on():
                        # Set up the timer wakeup
                        print(
                            f"THE DEVICE IS ON FOR TOO LONG: {time.time() - esp32_device.get_epoch_time_at_subscription()} seconds.")
                        sleep_time_seconds = 1  # in seconds
                        simulate_deep_sleep(client, sleep_time_seconds)

                if esp32_device.get_subscribed_flag() is False:
                    print("if esp32_device.subscribed_flag is False")
                    if time.time() - esp32_device.get_epoch_time_at_subscription() >= esp32_device.get_max_time_for_successful_subscription():
                        # Set up the timer wakeup
                        print(
                            f"THE DEVICE IS ON BUT NOT SUBSCRIBED FOR TOO LONG: {time.time() - esp32_device.get_epoch_time_at_subscription()} seconds.")
                        sleep_time_seconds = 40  # in seconds
                        simulate_deep_sleep(client, sleep_time_seconds)

                # CODICE PRINCIPALE QUI
                print_debug("esp32_device.subscribed_flag: ")
                print(esp32_device.get_subscribed_flag())
                print_debug("esp32_device.start_recording_now: ")
                print(esp32_device.get_start_recording_now())
                print_debug("esp32_device.start_sending_audio_now: ")
                print(esp32_device.get_start_sending_audio_now())
                print_debug("esp32_device.currently_sending_audio: ")
                print(esp32_device.get_currently_sending_audio())
                print_debug("esp32_device.new_audio_data_is_available: ")
                print(esp32_device.get_new_audio_data_is_available())

                # SENDING AUDIO
                if esp32_device.get_start_sending_audio_now() is True:
                    print_debug("if esp32_device.start_sending_audio_now is True) ")

                    local_time = time.localtime(time.time())
                    # Format struct_time to a string
                    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
                    print_debug(f"formatted_time: {formatted_time}")

                    # Send audio
                    print("+++++++++++++++++++++++++")
                    print("Sending audio")

                    esp32_device.update_currently_sending_audio(True)
                    simulate_publish_audio(client, esp32_device)
                    esp32_device.update_currently_sending_audio(False)  # we are no longer sending the audio
                    esp32_device.update_new_audio_data_is_available(
                        False)  # since we sent the last audio recorded, a new audio data must be recorded
                    esp32_device.update_start_recording_now(
                        False)  # no more recordings, the device should go on deep sleep mode
                    esp32_device.update_start_sending_audio_now(False)

                # RECORDING AUDIO
                if esp32_device.get_start_recording_now() is True and esp32_device.get_currently_sending_audio() is False and esp32_device.get_new_audio_data_is_available() is False:
                    print(
                        "if esp32_device.start_recording_now is True && esp32_device.currently_sending_audio is False)")
                    esp32_device.update_start_recording_now(False)

                    # RECORDING
                    # Create the I2S RECORD task and store the handle
                    print("RECORDING:  ")

                    simulate_record_and_notify(client, 20, esp32_device)

    except Exception as e:
        print_debug(f"An error occurred in main(): {e}")
        client.loop_stop()  # Ensure the loop is stopped
        client.disconnect()  # Ensure the client is disconnected


if __name__ == "__main__":
    while True:
        main()  # If the main finishes or fails, restart it
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("                                                               ")
        print("                                                               ")
        print("                                                               ")
        print("               Restarting the main function...                 ")
        print("                                                               ")
        print("                                                               ")
        print("                                                               ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        set_device_default_values(esp32_device)