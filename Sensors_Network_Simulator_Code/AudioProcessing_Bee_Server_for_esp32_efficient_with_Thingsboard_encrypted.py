#!/usr/bin/env python
import base64
import time
import credentials
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import requests # for telegram bot


########################################################################################################################
# TELEGRAM CREDENTIALS
BOT_TOKEN = credentials.BOT_TOKEN
CHAT_ID = credentials.CHAT_ID
'''
telegram bot token for BeehiveQueenbot
7696183846:AAFu_XUB5mOtdK_TSMyEhWGfFSQIizWByMk
Start a conversation in your chat, and then paste this URL on the browser:
https://api.telegram.org/bot7696183846:AAFu_XUB5mOtdK_TSMyEhWGfFSQIizWByMk/getUpdates
Then check for the chat id in the json content, and save it:
chat id: 258970901
'''

########################################################################################################################
# THINGSBOARD CREDENTIALS
THINGSBOARD_HOST = credentials.THINGSBOARD_HOST
#THINGSBOARD_HOST = 'demo.thingsboard.io'
ACCESS_TOKEN = credentials.ACCESS_TOKEN  # Token provided by ThingsBoard after creating the device
THINGSBOARD_PORT = 1883
THINGSBOARD_KEEPALIVE = 40


########################################################################################################################
# MQTT CREDENTIALS
broker = "test.mosquitto.org"  # Change as necessary
# broker = "broker.emqx.io"  # Change as necessary
# broker = "demo.thingsboard.io"  # Change as necessary
port = 1883  # Common port for MQTT or 8883 for TLS
keepalive = 40
timeout = 20

# Specify the topics to subscribe to
topics = [
    ("Raspberry_connect_to_sensors_net/request", 2),
    ("Raspberry_connect_to_sensors_net/ack", 2),
    ("Raspberry_recording_problems", 2),
    ("Raspberry_get_infos", 2),
    ("receive_audio_test", 2),
    ("Raspberry_audio_available", 2)
]


########################################################################################################################


start_time = time.time()  # Record the start time
import ssl
import random
import math
import string
import json
import numpy as np

seed = 2018
np.random.seed(seed)
import os
import re # regex to extract substrings easily from a string
import numpy as np
import librosa
import soundfile as sf
import sys
import subprocess  # to call C++ executables
import threading

import shutil  # to move files from a folder to another
##############################################################
import paho.mqtt.client as mqtt
import logging

logging.basicConfig(level=logging.DEBUG)

DEBUG_MODE = True  # if False, there will be no prints

simulation_esp32_flag = True




def print_debug(string_to_print):
    if DEBUG_MODE is True:
        print(string_to_print)


print(f"Python version: {sys.version}")
print(f"Python version info: {sys.version_info}")
print(f"NumPy version: {np.__version__}")




class RaspberryPi:

    def __init__(self, device_model, device_id, devices_fixed_id_dict, num_current_devices, num_max_allowed_devices,
                 devices_infos_dict, start_time_sending_audio_request, list_of_devices_to_put_to_sleep, list_of_devices_hives_to_put_to_sleep,
                 max_waiting_time_for_receiving_audio, max_resending_request, time_between_a_series_of_audio_requests, last_audio_request_publish_time,
                 net_password, last_communication_epoch_dict, I2S_sample_rate, record_time, I2S_channel_num, I2S_sample_bits, max_hive_number_ever_registered,
                 max_time_without_communication_with_a_device, esp32_max_time_on, esp32_deep_sleep_time, update_thingsboard_led_status,
                 current_timer, subscription_in_progress_timer=0, telegram_notifications=False, dev_is_connected=False, receive_audios_folder_path="received_audio_files/",
                 collect_audios_folder_path="audio_files_collector/", audio_data_topic_key=None, communication_id=None,
                 current_invitation_id=None, ip_address=None, imminent_audio_request_led_flag_thingsboard=False,
                 devices_to_make_an_audio_request_list=None, go_on_with_audio_requests=None, caution_time_period_before_audio_request=40,
                 counter_resend_request_done=0, current_handled_device_id=None, subscription_in_progress=False,
                 split_3sec_audio_flag=False):
        """
        Initialize the Raspberry Py with relevant information.

        :param device_model: str - Model of the Raspberry Py
        :param device_id: str - Unique identifier for the Raspberry Py
        :param devices_fixed_id_dict: dict - dictionary containing as keys the fixed_ids of the devices that are part of the sensors net, and as values the related device_ids in the net
        :param num_current_devices: int - number of current devices making part of the sensors network
        :param num_max_allowed_devices: int - max number of allowed devices in the sensors network
        :param num_max_pending_subscription_requests: int - max number pending requests of devices that ask for being part of the sensors network (max dimension allowed for 'invitation_id_list')
        :param state: str - Current state of the Raspberry Py (e.g., "idle", "active", "sleeping", etc.)
        :param ip_address: str - IP address of the Raspberry Py in the network
        :param max_waiting_time_for_receiving_audio: int - max amount of time in seconds to wait for an audio to be sent from the ESP32 towards the RaspberryPi
        :param start_time_sending_audio_request: int - time in seconds epoch unit when the request for audio data has been sent from the RaspberryPi to the ESP32
        :param audio_data_topic_key: str - random generated key to be appended to "audio/payload" (thus obtaining "audio/payload"+audio_data_topic_key) in order to create a topic on fly and use it by the ESP32 to send the audio data towards the RaspberryPi
        :param communication_id: str - random generated id to use in order to be recognizable in a new or an already started communication with the ESP32, for a certain topic
        :param time_between_a_series_of_audio_requests: int - amount of time it should elapse between two consecutive series of audio requests (in seconds), where a series of audio requests involves a request done to all the active devices
        :param devices_infos_dict: dict - dictionary containing the infos of all the net's devices in json format, like: {device_id_1:{"device_model":<dev_model>, "location":<location>, ...}, ..., device_id_n":{...}}
        :param current_invitation_id: string - invitation_id of the current request to be part of the network still in progress
        :param invitation_id_list: list - queue list of the several invitation_id related to the devices waiting for a place in the sensors network. As soon as a place will be available, the first pending invitation_id of the queue will be used to invite the related device waiting to be part of the net.
        :param devices_to_make_an_audio_request_list: list - it's a list of the devices that already notified they have an audio ready to send.
               Whenever a device sends a notification to the RaspberryPi, it's device_id will be pushed into the list. Vice versa, whenever an audio has been successfully received, the related device_id will be popped out by the list.
        #  NO MORE   :param start_index_ready_devices_list: int - it's the start_index for devices_to_make_an_audio_request_list that identifies a group pf device_ids to request for an audio.
                E.g., if we have: devices_to_make_an_audio_request_list = [ID1, ID2, ID3, ID4, ID5, ID6, ID7, ID8, ID9, ID10, ID11, ID12, ID13, ID14], then start_index will have value:
                - 0 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPi is quad-core) to the devices with ids: ID1, ID2, ID3, ID4, since this group starts from index 0;
                - 4 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPi is quad-core) to the devices with ids: ID5, ID6, ID7, ID8, since this group starts from index 4;
                - 8 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPi is quad-core) to the devices with ids: ID9, ID10, ID11, ID12, since this group starts from index 8;
                - 12 --> so that we can request an audio in parallel (4 parallel requests since the RaspberryPi is quad-core) to the devices with ids: ID13, ID14, since this group starts from index 12;
                So, after every parallel audio requests (made of max requests) and the consecutive audio saving, start_index_ready_devices_list will be increased for the next parallel audio request, and so on (after all the IDs have been interrogated, the variable will be set to 0)
        #  NO MORE   :param threads_communication_id_dict: dict - dictionary containing the communication_ids created in the multithreaded audio request. Since these must be used also in the audio/ack topic, it's better to save them for that occurrence as a class field
        #  NO MORE   :param thread_audio_data_topic_key_dict: dict - dictionary containing the audio_data_topic_keys created in the multithreaded audio request. Since these must be used also in the audio/payload topic, it's better to save them for that occurrence as a class field
        :param audio_data_topic_key: string - audio_data_topic_key created right before the audio request. Since this must be used also in the audio/payload topic, it's better to save it for that occurrence as a class field
        #  NO MORE   :param go_on_with_audio_requests_dict: dict - dict of key:values pairs like threadx:True/False, where if True it means that it is possible to make the next request for that thread (the previous request has been successfully handled), otherwise not if False
                When it's time to make the next multithreaded requests, every value of the dict must be True, so a check on these is necessary.
        :param go_on_with_audio_requests: boolean - (True/False), where if True it means that it is possible to make the next request (the previous request has been successfully handled), otherwise not if False
        :param max_resending_request: int - the maximum number of audio resend request allowed (if an audio is not well received for the max time allowed, then the problem must be ignored sending an ack:"ok" to the device)
        #  NO MORE   :param counters_resend_request_done_per_thread_dict: dict - a dictionary containing, for every of the four threads, the counters of the resending requests, in order to take trace of it if a counter becomes >= than the max_resending_request number
        :param counter_resend_request_done: int - the counters of the resending requests, in order to take trace of it if the counter becomes >= than the max_resending_request number
        #  NO MORE   :param current_device_id_handled_by_thread_dict: dict - a dictionary containing as keys the four threads (e.g. thread1) and for each of them the value is the current device_id handled by that thread at the moment (assigned during the audio request phase)
        :param current_handled_device_id: string - the current device_id handled at the moment (assigned during the audio request phase)
        :param net_password: string - password used only the very first time trying to ask to the RaspberryPi for a place in the sensors network
        :param subscription_in_progress: boolean - (True/False) if True, there is a subscription to the sensors network still in progress, and no other devices can ask for a subscription till the flag is True. A subscription request can be done only if the flag is False.
        :param receive_audios_folder_path: string - folder path where to receive the audios sent from the several devices through the current series of audio requests
        :param collect_audios_folder_path: string - folder path where to collect the audios sent from the several devices through all the audio requests (after a series of audio requests, the related audios will be moved from the receive_audios_folder_path to collect_audios_folder_path)
        :param dev_is_connected: boolean - when we do an audio request, it could happen that the device we sent the request to is no more connected to the mqtt network...thus there is the risk to wait for an audio indefinitely. To avoid this situation,
        we request the state infos to the devices in order to know if the dev is responsive in the net. If it is, we'll receive these infos, and we set the variable 'dev_is_connected' to True, otherwise to False.
        After this check, we set the variable 'dev_is_connected' to False, for other possible successive audio requests.
        :param last_communication_epoch_dict: dict - dictionary containing as keys the device_ids, and as their values the related last_communication_epoch (that is the last time we sent that device, expressed as epoch)
        :param max_time_without_communication_with_a_device: int - it's the max time that could elapses for a device to not communicate with the raspberry
        :param split_3sec_audio_flag: bool - when True, the esp32 devices will record 3sec audios that will be split from the raspberrypy in 1 sec audios
        :param I2S_sample_rate: int - sample rate for the esp32 devices
        :param record_time: int - time duration (in seconds) of an audios recorded by the esp32 devices
        :param I2S_channel_num: int - num of channels for the audios recorded by the esp32 devices
        :param I2S_sample_bits: int - bit depth of the audios recorded by the esp32 devices
        :param esp32_max_time_on: int - the esp32 cannot stay on more than this amount of seconds (ON and then deep_sleep)
        :param esp32_deep_sleep_time: int - the esp32 should go in deep_sleep mode for this amount of seconds
        :param list_of_devices_to_put_to_sleep: int - list of hive numbers to be put to sleep (the hive number is contained in the 'location')
        :param update_thingsboard_led_status: bool - if True it means the thingsboard led status must be updated after a device subscription
        :param current_timer: int - current timer that goes from 0 to till the audio request time, and then returns 0 (if there are not devices ready to sent an audio, the timer goes on and on...)
        :param telegram_notifications: bool - if it's True, every time is detected a 'noqueen' state for any hive, a telegram notification will be sent to the chat
        :param imminent_audio_request_led_flag_thingsboard: bool - if it's True, in about 40sec the aspberry will do the aduio requests to the esp32 connected, and the led on the Thingsboard Dashboard should turn on (orange)
        :param caution_time_period_before_audio_request: int - the period of time that starts "caution_time_period_before_audio_request" seconds before the audio request is considered a period of time when
               the several esp32 try to re-subscribe to the sensors network, so eventually generating a lot of mqtt traffic (thus it is suggested to not give other commands to the raspberry py at this moment)
        :param last_audio_request_publish_time: float - last epoch related to the moment we finished handling the last audio request (so at the very end of it, right after the classification)
        :param subscription_in_progress_timer: int - it could happen that an already started subscription attempt doesn't come to an end... So the flag subscription_in_progress could stay True,
               leading to the inability for other devices to subscribe (since they could subscribe only if subscription_in_progress = False). To solve this problem, subscription_in_progress_timer will be 0
               whenever a subscription attempt starts, and after 15 seconds it will be put automatically to subscription_in_progress = False, in order for eventual other devices to subscribe successfully
        :param max_hive_number_ever_registered: int - this utility is usefull for thingsboard: if we add an hive in the system, with a desired number, and then at a certain point we disconnect it, its led
               doesn't turn off since we deleted it from the sensors network and we lost the information about its hive number --> we don't know which led number to turn Off...So, whenever an hive
               registers to the sensors network, everytime we have to update the dashboard's leds, we check which is the max hive number among all the hive numbers,
               and then we upload this value in this class's field to reuse it in order to turn off every led that has already been deleted, that is present and inside the range [1 : max_hive_number_ever_registered]
        """

        self.device_model = device_model
        self.device_id = device_id
        self.devices_fixed_id_dict = devices_fixed_id_dict  # e.g. {fixed_id1: device_id1, fixed_id2: device_id2, ...}
        self.num_current_devices = num_current_devices
        self.num_max_allowed_devices = num_max_allowed_devices
        # self.num_max_pending_subscription_requests = num_max_pending_subscription_requests
        # self.state = state
        self.ip_address = ip_address
        self.start_time_sending_audio_request = start_time_sending_audio_request
        self.max_waiting_time_for_receiving_audio = max_waiting_time_for_receiving_audio
        self.communication_id = communication_id
        self.time_between_a_series_of_audio_requests = time_between_a_series_of_audio_requests
        self.devices_infos_dict = devices_infos_dict
        # self.invitation_id_list = invitation_id_list  # e.g. [invitation_id1, invitation_id2, invitation_id3, invitation_id4, ...]
        self.current_invitation_id = current_invitation_id
        self.devices_to_make_an_audio_request_list = devices_to_make_an_audio_request_list  # e.g. [device_id13, device_id4, device_id5, device_id11, ...]
        # NON SERVE PIÙ PERCHÈ NON VENGONO FATTE PIÙ RICHIESTE MQTT IN MULTITHREADING - self.start_index_ready_devices_list = start_index_ready_devices_list #e.g. can be one of the following: 0/4/8/12/16/20....see the example above in the comment section of this class
        # USO DIRETTAMENTE communication_id - self.threads_communication_id_dict = threads_communication_id_dict # e.g. {"thread1": communication_id1, "thread2": communication_id2, "thread3": communication_id3, "thread4": communication_id4}
        # NON SERVE PIÙ PERCHÈ NON VENGONO FATTE PIÙ RICHIESTE MQTT IN MULTITHREADING - self.thread_audio_data_topic_key_dict = thread_audio_data_topic_key_dict # e.g. {"thread1": topic_key1, "thread2": topic_key2, "thread3": topic_key3, "thread4": topic_key4}
        self.audio_data_topic_key = audio_data_topic_key  # e.g. audio_data_topic_key = topic_key
        # VA USATO SOLO UN FLAG PER UNA SOLA RICHIESTA INVECE CHE UN DIZIONARIO (VEDI go_on_with_audio_requests) - self.go_on_with_audio_requests_dict = go_on_with_audio_requests_dict # e.g. {"thread1": True, "thread2": False, "thread3": True, "thread4": False}
        self.go_on_with_audio_requests = go_on_with_audio_requests  # e.g. go_on_with_audio_requests = True
        self.max_resending_request = max_resending_request  # e.g. 3 (3 attempts to resend an audio)
        # VA USATO SOLO UN CONTATORE (vedi counter_resend_request_done) - self.counters_resend_request_done_per_thread_dict = counters_resend_request_done_per_thread_dict # e.g. {"thread1": 0, "thread2": 2, "thread3": 0, "thread4": 1}
        self.counter_resend_request_done = counter_resend_request_done  # e.g. counter_resend_requests_done = 2
        # VA USATO SOLO UN CURRENT HANDLED DEVICE ID (current_handled_device_id) -  self.current_device_id_handled_by_thread_dict = current_device_id_handled_by_thread_dict # e.g. {"thread1": device_id13, "thread2": device_id4, "thread3": device_id5, "thread4": device_id11}
        self.current_handled_device_id = current_handled_device_id  # e.g. current_handled_device_id = device_id4
        self.net_password = net_password
        self.subscription_in_progress = subscription_in_progress
        self.receive_audios_folder_path = receive_audios_folder_path
        self.collect_audios_folder_path = collect_audios_folder_path
        self.dev_is_connected = dev_is_connected  # e.g. dev_is_connected = True/False
        self.last_communication_epoch_dict = last_communication_epoch_dict  # e.g. {device_id1: last_communication_epoch1, device_id2: last_communication_epoch2, ...}
        self.max_time_without_communication_with_a_device = max_time_without_communication_with_a_device
        self.split_3sec_audio_flag = split_3sec_audio_flag
        self.I2S_sample_rate = I2S_sample_rate
        self.record_time = record_time
        self.I2S_channel_num = I2S_channel_num
        self.I2S_sample_bits = I2S_sample_bits
        self.esp32_max_time_on = esp32_max_time_on
        self.esp32_deep_sleep_time = esp32_deep_sleep_time
        self.list_of_devices_to_put_to_sleep = list_of_devices_to_put_to_sleep
        self.list_of_devices_hives_to_put_to_sleep = list_of_devices_hives_to_put_to_sleep
        self.update_thingsboard_led_status = update_thingsboard_led_status
        self.current_timer = current_timer
        self.telegram_notifications = telegram_notifications
        self.imminent_audio_request_led_flag_thingsboard = imminent_audio_request_led_flag_thingsboard
        self.caution_time_period_before_audio_request = caution_time_period_before_audio_request
        self.last_audio_request_publish_time = last_audio_request_publish_time
        self.subscription_in_progress_timer = subscription_in_progress_timer
        self.max_hive_number_ever_registered = max_hive_number_ever_registered

    def get_hive_number_from_location(self, location):
        print_debug("get_thingsboard_state_value")
        # location = self.devices_infos_dict[device_id]['location']
        numbers = re.findall(r'\d+', location)
        return int(numbers[0])

    def get_thingsboard_values(self, device_id):
        print_debug("get_thingsboard_state_value")
        location = self.devices_infos_dict[device_id]['location']
        state = self.devices_infos_dict[device_id]['state']
        thingsboard_hive = "Hive1"
        thingsboard_state = 0
        # Find all numbers in the string
        numbers = re.findall(r'\d+', location)

        # If you expect only one number and want an integer
        if numbers:
            hive_number = int(numbers[0])  # Convert the first found number to an integer
            print(f"hive_number:{hive_number}")
            thingsboard_hive = "Hive"+str(hive_number)
            '''
            On Thingsboard, every hive has its own graph in the same shared Dashboard (a square wave, where the On state is represented by the queen_value, and viceversa the Off state is the noqueen_value)
            How to obtain the thingsboard_value for each hive, starting from the hive_number:
            
            Hive        noqueen     queen       noqueen             queen
                                                formula             formula
            ----------------------------------------------------------------
            Hive1       0           1           1*2 - 2 = 0         0+1 = 1
            Hive2       2           3           2*2 - 2 = 2         2+1 = 3
            Hive3       4           5           3*2 - 2 = 4         4+1 = 5
            Hive4       6           7           4*2 - 2 = 6         6+1 = 7
            Hive5       8           9           5*2 - 2 = 8         8+1 = 9
            Hive6       10          11          6*2 - 2 = 10        10+1 = 11
            Hive7       12          13          7*2 - 2 = 12        12+1 = 13
            Hive8       14          15          8*2 - 2 = 14        14+1 = 15
            .
            .
            .
            Hive20      38          39          20*2- 2 = 38        38+1 = 39
            
            '''

            noqueen_value = hive_number * 2 - 2
            queen_value = noqueen_value + 1

            if state == "noqueen":
                thingsboard_state = noqueen_value
            elif state == "queen":
                thingsboard_state = queen_value

        return thingsboard_hive, thingsboard_state

    def get_list_of_hives_numbers_to_put_to_sleep(self):
        print_debug("get_list_of_hives_numbers_to_put_to_sleep")
        list_of_hives_numbers_to_put_to_sleep = []
        for element in self.list_of_devices_hives_to_put_to_sleep:
            for device_id, hive in element.items(): # list_of_devices_hives_to_put_to_sleep is like: [{device_id1: hive_number1}, ..., {device_idN: hive_numberN}]
                list_of_hives_numbers_to_put_to_sleep.append(hive)
        print(f"list_of_hives_numbers_to_put_to_sleep: {list_of_hives_numbers_to_put_to_sleep}")
        return list_of_hives_numbers_to_put_to_sleep

    def get_list_of_devices_id_to_put_to_sleep(self):
        print_debug("get_list_of_devices_id_to_put_to_sleep")
        list_of_devices_to_put_to_sleep = []
        for element in self.list_of_devices_hives_to_put_to_sleep: # list_of_devices_hives_to_put_to_sleep is like: [{device_id1: hive_number1}, ..., {device_idN: hive_numberN}]
            for device_id in element:
                list_of_devices_to_put_to_sleep.append(device_id)
        print(f"list_of_devices_to_put_to_sleep: {list_of_devices_to_put_to_sleep}")
        return list_of_devices_to_put_to_sleep

    def remove_from_list_of_devices_hives_to_put_to_sleep_given_the_hive_number(self, hive_number):
        print_debug("remove_from_list_of_devices_hives_to_put_to_sleep_given_the_hive_number")
        for element in self.list_of_devices_hives_to_put_to_sleep:
            for device_id, hive_num in element.items(): # list_of_devices_hives_to_put_to_sleep is like: [{device_id1: hive_number1}, ..., {device_idN: hive_numberN}]
                if hive_number == hive_num:
                    self.list_of_devices_hives_to_put_to_sleep.remove(element)
                    break
        print(f"self.list_of_devices_hives_to_put_to_sleep: {self.list_of_devices_hives_to_put_to_sleep}")

    def remove_from_list_of_devices_hives_to_put_to_sleep_given_the_device_id(self, device_id):
        print_debug("remove_from_list_of_devices_hives_to_put_to_sleep_given_the_device_id")
        for element in self.list_of_devices_hives_to_put_to_sleep:  # list_of_devices_hives_to_put_to_sleep is like: [{device_id1: hive_number1}, ..., {device_idN: hive_numberN}]
            for dev_id in element:
                if device_id == dev_id:
                    self.list_of_devices_hives_to_put_to_sleep.remove(element)
                    break
        print(f"self.list_of_devices_hives_to_put_to_sleep: {self.list_of_devices_hives_to_put_to_sleep}")



    def delete_device_from_the_net(self, device_id):
        print_debug(f"- - - - - - - - - delete_device_from_the_net --> DELETED DEVICE WITH DEVICE_ID: {device_id}")
        print("BEFORE DELETING:")
        print(f"num_current_devices: {self.num_current_devices}")
        print(f"devices_fixed_id_dict: {self.devices_fixed_id_dict}")
        print(f"devices_infos_dict: {self.devices_infos_dict}")
        print(f"last_communication_epoch_dict: {self.last_communication_epoch_dict}")
        fixed_id_to_delete = ""
        for fixed_id in self.devices_fixed_id_dict:
            if self.devices_fixed_id_dict[fixed_id] == device_id:
                fixed_id_to_delete = fixed_id
                break
        # delete from devices_fixed_id_dict
        del self.devices_fixed_id_dict[fixed_id_to_delete]
        # delete from last_communication_epoch_dict
        del self.last_communication_epoch_dict[device_id]
        # decrease by 1 the number of current devices in the net
        self.num_current_devices -= 1
        if device_id in self.devices_infos_dict:
            # delete from devices_infos_dict
            del self.devices_infos_dict[device_id]
        print("AFTER DELETING:")
        print(f"num_current_devices: {self.num_current_devices}")
        print(f"devices_fixed_id_dict: {self.devices_fixed_id_dict}")
        print(f"devices_infos_dict: {self.devices_infos_dict}")
        print(f"last_communication_epoch_dict: {self.last_communication_epoch_dict}")


    def update_beehive_state(self, device_id, state):
        print_debug("update_beehive_state")
        if device_id in self.devices_infos_dict:
            # Update the state of the device
            self.devices_infos_dict[device_id]['state'] = state
            print(f"Updated state of device {device_id} to {state}.")
        else:
            print(f"Device ID {device_id} not found.")

    def update_last_communication_epoch_dict(self, device_id):
        print_debug("update_last_communication_epoch_dict")
        epoch_time = time.time()
        self.last_communication_epoch_dict.update({device_id: epoch_time})

    def append_device_with_ready_audio_to_send_list(self, device_id):
        print_debug("append_device_with_ready_audio_to_send_list")
        if device_id not in self.devices_to_make_an_audio_request_list:
            self.devices_to_make_an_audio_request_list.append(device_id)
            print_debug(f"self.devices_to_make_an_audio_request_list: {self.devices_to_make_an_audio_request_list}")

    def update_devices_infos_dict(self, json_received_infos):
        print("update_devices_infos_dict")
        print_debug(f"before DEL json_received_infos: {json_received_infos}")
        device_id = json_received_infos["device_id"]
        print_debug(f"device_id: {device_id}")
        del json_received_infos[
            "device_id"]  # deleting the key:value with key "device_id", so that it remains all the dict without it, and we can use it as a 'value' in the update
        print_debug(f"after DEL json_received_infos: {json_received_infos}")
        # self.devices_infos_dict[device_id] = json_received_infos
        # print_debug("self.devices_infos_dict: ", self.devices_infos_dict)
        self.devices_infos_dict.update({device_id: json_received_infos})
        print(f"self.devices_infos_dict: {self.devices_infos_dict}")
        '''
        device_infos = {
        "device_model" : json_received_infos["device_model"],
        "location" : json_received_infos["location"],
        "battery_level" : json_received_infos["battery_level"],
        "state" : json_received_infos["state"],
        "ip_address" : json_received_infos["ip_address"],
        "enable_audio_record" : json_received_infos["enable_audio_record"],
        "last_record_date" : json_received_infos["last_record_date"],
        }
        self.devices_infos_dict.update({device_id: device_infos})
        '''

def generate_random_code(length):
    print_debug("generate_random_code")
    # Create a pool of uppercase, lowercase letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random code by selecting random characters from the pool
    random_code = ''.join(random.choice(characters) for _ in range(length))
    return random_code


def is_wav_file(data):
    print_debug("is_wav_file")
    # Check if the first four bytes are the b'RIFF' and the next four are b'WAVE'
    return len(data) > 8 and data[0:4] == b'RIFF' and data[8:12] == b'WAVE'

def publish_response_to_subscription_request(client, topic, invitation_id, answer, device_id, RaspberryPi):
    response = {}
    if answer == "allowed":
        print_debug("elif answer == allowed:")
        response = {"device_id": device_id, "invitation_id": invitation_id, "Connection_permission": "allowed",
                    "deep_sleep_time": RaspberryPi.esp32_deep_sleep_time, "max_time_on": RaspberryPi.esp32_max_time_on,
                    "I2S_sample_rate": RaspberryPi.I2S_sample_rate, "record_time": RaspberryPi.record_time,
                    "I2S_channel_num": RaspberryPi.I2S_channel_num, "I2S_sample_bits": RaspberryPi.I2S_sample_bits, "current_timer": RaspberryPi.current_timer}
    elif answer == "denied":
        print_debug("elif answer == denied:")
        response = {"invitation_id": invitation_id, "Connection_permission": "denied", "deep_sleep_time": RaspberryPi.esp32_deep_sleep_time,
                    "max_time_on": RaspberryPi.esp32_max_time_on}
    elif answer == "already_in":
        print_debug("elif answer == already_in:")
        response = {"device_id": device_id, "invitation_id": invitation_id, "Connection_permission": "already_in",
                    "deep_sleep_time": RaspberryPi.esp32_deep_sleep_time, "max_time_on": RaspberryPi.esp32_max_time_on,
                    "I2S_sample_rate": RaspberryPi.I2S_sample_rate, "record_time": RaspberryPi.record_time,
                    "I2S_channel_num": RaspberryPi.I2S_channel_num, "I2S_sample_bits": RaspberryPi.I2S_sample_bits, "current_timer": RaspberryPi.current_timer}

    response = json.dumps(response)
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug("+++++++++++++++++++++++++++++")
    print_debug(f"formatted_time: {formatted_time}")
    print_debug(f"topic: {topic}")
    print_debug(f"Sending: {response}")
    encrypted_response = encrypt(response)
    client.publish(topic, encrypted_response, qos=2) # client


def publish_audio_request(client, topic, device_id, communication_id, audio_data_topic_key):
    print_debug("publish_audio_request")
    request = {"device_id": device_id, "audio_data_topic_key": audio_data_topic_key,
               "communication_id": communication_id}
    request = json.dumps(request)
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    print_debug("+++++++++++++++++++++++++++++")
    print_debug(f"formatted_time: {formatted_time}")
    print_debug(f"topic: {topic}")
    print_debug(f"Sending: {request}")
    encrypted_request = encrypt(request)
    client.publish(topic, encrypted_request, qos=2) # client

    '''
    topic: audio/request
    ESP32                                                                                                RASPBERRYPY
    _____                                                                                                ______
    # 1      {device_id:<ID>, audio_data_topic_key:<topic_key>, communication_id:<ID>}<---------------------

    Dynamic topic: audio/payload<key>
    ESP32                                                                                                RASPBERRYPY
    _____                                                                                                ______
    # 1      -------------------------------------------------------------------------------------------->audio_content
    '''


def request_audio(client, RaspberryPi, device_id, audio_data_topic_key):
    print_debug("request_audio")
    RaspberryPi.audio_data_topic_key = audio_data_topic_key
    print_debug(f"RaspberryPi.audio_data_topic_key: {RaspberryPi.audio_data_topic_key}")
    communication_id = generate_random_code(20)
    RaspberryPi.communication_id = communication_id
    client.subscribe("audio/payload" + audio_data_topic_key, qos=2) # client
    publish_audio_request(client, "audio/request", device_id, communication_id, audio_data_topic_key)


def ack_audio_received(client, RaspberryPi, ack):
    print_debug("ack_audio_received")
    # audio/ack:
    # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
    # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
    if ack == "resend":
        # 3      {communication_id:<ID>, audio_data_topic_key:<topic_key>, ack:"resend"}<-----------------------------------
        audio_data_topic_key = generate_random_code(19) + '-'  # must be different than the previous one
        topic_where_to_get_the_audio = "audio/payload" + audio_data_topic_key
        client.subscribe(topic_where_to_get_the_audio, qos=2) # client
        time.sleep(1)
        RaspberryPi.audio_data_topic_key = audio_data_topic_key
        communication_id = RaspberryPi.communication_id
        message = {"communication_id": communication_id, "audio_data_topic_key": audio_data_topic_key, "ack": ack}
        message = json.dumps(message)
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug("+++++++++++++++++++++++++++++")
        print_debug(f"formatted_time: {formatted_time}")
        print_debug(f"topic: audio/ack")
        print_debug(f"Sending: {message}")
        encrypted_message = encrypt(message)
        client.publish("audio/ack", encrypted_message, qos=2) # client
    elif ack == "ok":
        # 3      {communication_id:<ID>, ack:"ok"}<-----------------------------------
        print_debug(f"RaspberryPi.communication_id: {RaspberryPi.communication_id}")
        communication_id = RaspberryPi.communication_id
        message = {"communication_id": communication_id, "ack": ack}
        message = json.dumps(message)
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug("+++++++++++++++++++++++++++++")
        print_debug(f"formatted_time: {formatted_time}")
        print_debug(f"topic: audio/ack")
        print_debug(f"Sending: {message}")
        encrypted_message = encrypt(message)
        client.publish("audio/ack", encrypted_message, qos=2) # client



def save_audio_from_payload(client, msg, raspberrypi3B):
    print_debug("save_audio_from_payload")
    go_on_with_audio_request = False

    if is_wav_file(msg.payload):
        # os.makedirs(f"received_audio_files/thread{thread_number}", exist_ok=True)

        device_id = raspberrypi3B.current_handled_device_id
        device_model = raspberrypi3B.devices_infos_dict[device_id]["device_model"]
        location = raspberrypi3B.devices_infos_dict[device_id]["location"]

        raspberrypi3B.update_last_communication_epoch_dict(device_id)

        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)

        last_record_date = formatted_time
        if simulation_esp32_flag is False:
            file_name = f"{device_model}_({location})_{last_record_date}"
        else:
            random_label = random.choice(["queen", "noqueen"])
            file_name = f"{device_model}_({location})_{last_record_date}_{random_label}"

        # with open(f"received_audio_files/thread{thread_number}/{file_name}.wav", "wb") as audio_file:
        # audio_data = base64.b64decode(msg.payload.decode('utf-8')) #*****************
        with open(f"{raspberrypi3B.receive_audios_folder_path}{file_name}.wav", "wb") as audio_file:
            # audio_file.write(audio_data)
            audio_file.write(msg.payload)
        # print_debug(f"WAV audio file saved as received_audio_files/thread{thread_number}/{file_name}.wav")
        print_debug(f"WAV audio file saved as received_audio_files/{file_name}.wav")
        go_on_with_audio_request = True
        raspberrypi3B.audio_data_topic_key = None
    else:
        print_debug("Received file is not a valid .wav file.")
        raspberrypi3B.audio_data_topic_key = None
        go_on_with_audio_request = False

    if go_on_with_audio_request is True:
        print_debug("if go_on_with_audio_request is True:")
        ack_audio_received(client, raspberrypi3B, "ok")  # ack ok (since everything went fine)
    else:
        print_debug("else:")
        if raspberrypi3B.counter_resend_request_done < raspberrypi3B.max_resending_request:  # retry asking the same audio
            print_debug(
                "if raspberrypi3B.counters_resend_request_done_per_thread_dict[fthread{thread_number}] < raspberrypi3B.max_resending_request: # retry asking the same audio")
            ack_audio_received(client, raspberrypi3B, "resend")
            go_on_with_audio_request = False
        else:  # too many attempts, we don't care anymore, just say True and go on ...
            print_debug(
                "else: # too many attempts, we don't care anymore, just say True and go on ...")
            ack_audio_received(client, raspberrypi3B, "ok")
            go_on_with_audio_request = True

    raspberrypi3B.go_on_with_audio_requests = go_on_with_audio_request
    print_debug(f"raspberrypi3B.go_on_with_audio_requests after saving = {raspberrypi3B.go_on_with_audio_requests}")




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
        return True          # Return True if successful
    except json.JSONDecodeError:
        return False         # Return False if there is a parsing error





raspberrypi3B = RaspberryPi(
    device_model="RaspberryPi_3B",
    device_id="Rpi3B",
    devices_fixed_id_dict={},
    num_current_devices=0,
    num_max_allowed_devices=4,
    ip_address="192.168.1.200",
    start_time_sending_audio_request=None,
    max_waiting_time_for_receiving_audio=5,
    communication_id=None,
    esp32_max_time_on=260,
    esp32_deep_sleep_time=570,
    time_between_a_series_of_audio_requests=600,
    devices_infos_dict={},
    current_invitation_id=None,
    devices_to_make_an_audio_request_list=[],
    audio_data_topic_key=None,
    go_on_with_audio_requests=False,
    max_resending_request=3,
    counter_resend_request_done=0,
    current_handled_device_id=None,
    net_password=credentials.net_password,
    subscription_in_progress=False,
    receive_audios_folder_path="received_audio_files/",
    collect_audios_folder_path="audio_files_collector/",
    last_communication_epoch_dict={},
    max_time_without_communication_with_a_device=900,
    split_3sec_audio_flag=False,
    I2S_sample_rate=22050,
    record_time=1,
    I2S_channel_num=1,
    I2S_sample_bits=16,
    list_of_devices_to_put_to_sleep = [],
    list_of_devices_hives_to_put_to_sleep = [],
    update_thingsboard_led_status = False,
    current_timer=0,
    imminent_audio_request_led_flag_thingsboard=False,
    caution_time_period_before_audio_request=40,
    last_audio_request_publish_time = 0,
    max_hive_number_ever_registered = 20
)


def send_telegram_message(chat_id, message, token):
    print("send_telegram_message")
    print(f"message: {message}")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.json()



def on_log(client, userdata, level, buf):
    print_debug("on_log")
    print_debug(f"log: {buf}")


def on_disconnect(client, userdata, rc, properties=None):
    print_debug("on_disconnect")
    print(f"Disconnected with result code {rc}")
    if rc != 0:
        print("Unexpected disconnection. Trying to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"Reconnection failed: {e}")


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


# Helper function to determine if the received payload is the last chunk
def is_last_chunk(msg):
    # Implementation can vary. For example, you might check a specific property of the msg
    # or inspect the payload for a special marker.
    # Here you can implement your own logic based on your protocol
    return len(msg.payload) == 0  # Placeholder condition




###################################################################################
# THINGSBOARD MQTT

def on_log_thingsboard(client_thingsboard, userdata, level, buf):
    print_debug("on_log_thingsboard")
    print_debug(f"log: {buf}")


def on_disconnect_thingsboard(client_thingsboard, userdata, rc, properties=None):
    print_debug("on_disconnect_thingsboard")
    print(f"Disconnected with result code {rc}")
    if rc != 0:
        print("Unexpected disconnection. Trying to reconnect...")
        try:
            client_thingsboard.reconnect()
        except Exception as e:
            print(f"Reconnection failed: {e}")


# Callback function when the client receives a CONNACK response from the server.
def on_connect_thingsboard(client_thingsboard, userdata, flags, rc, properties=None):
    print_debug("on_connect_thingsboard")
    # print_debug(f"Connected to broker at {broker} with result code {rc}")
    logging.debug(f"Connected to broker at {THINGSBOARD_HOST} with result code {rc}")
    if rc == 0:
        print_debug("Connection successful!")
        client_thingsboard.subscribe("v1/devices/me/rpc/request/+", 1)
        print_debug(f"Subscribed to topic: 'v1/devices/me/rpc/request/+' with qos=1")
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


def on_message_thingsboard(client_thingsboard, userdata, msg):
    print_debug(f"on_message_thingsboard")
    data = json.loads(msg.payload)
    print("Received message:", msg.payload.decode())
    print("data:", data)

    '''
    DASHBOARD OBJECT                on_message method               on_message params               update_thingsboard_dashboard method         update params/values
    ______________________________________________________________________________________________________________________________________________________________________
    led sleep (10x):                    -                                   -                       ledStatus1, ..., ledStatus10                True(On)/False(Off)
    ______________________________________________________________________________________________________________________________________________________________________
    switch sleep (10x):             sleep_Hive/1,                   True(sleep)/False(awake)        state_sleep_Hive/1,                         True(sleep)/False(awake)
                                    ...,                                                                ...,
                                    sleep_Hive/10                                                       state_sleep_Hive/10
    ______________________________________________________________________________________________________________________________________________________________________
    led measure period (3x):                -                               -                       led10min, led30min, led1hour                True(On)/False(Off)
    ______________________________________________________________________________________________________________________________________________________________________
    button measure period (3x):     set_time_min/10,                        -                                   -                                           -
                                    set_time_min/30,                        -                                   -                                           -
                                    set_time_min/60                         -                                   -                                           -
    ______________________________________________________________________________________________________________________________________________________________________
    Alarms timeseries table (1x):           -                               -                       Hive_alarm,                                 HiveX string
                                            -                               -                       Device_model_alarm,                         Device-modelX string
                                            -                               -                       State_alarm                                 noqueen string
    ______________________________________________________________________________________________________________________________________________________________________
    Beehive stats charts (1x):              -                               -                       Hive1,                                      0(noqueen)/1(queen),
                                                                                                    Hive2,                                      2(noqueen)/3(queen),
                                                                                                    ...,                                        ...,
                                                                                                    Hive9,                                      16(noqueen)/17(queen),
                                                                                                    Hive10                                      18(noqueen)/19(queen)
    ______________________________________________________________________________________________________________________________________________________________________
    Telegram switch (1x)            set_state_Telegram_switch       True(On)/False(Off)             state_Telegram_switch                       True(On)/False(Off)
    ______________________________________________________________________________________________________________________________________________________________________
    '''



    if msg.topic.startswith('v1/devices/me/rpc/request/'):

        if data['method'].startswith("sleep_Hive/"):
            print_debug("if data['method'].startswith('sleep_Hive/'):")
            splitted_topic = data['method'].split('/')
            hive_number = int(splitted_topic[-1])
            print(f"hive_number: {hive_number}")
            '''
            list_of_devices = []
            for device_id, info in raspberrypi3B.devices_infos_dict.items():
                location = raspberrypi3B.devices_infos_dict[device_id]['location']
                hive_number = raspberrypi3B.get_hive_number_from_location(location)
                list_of_devices.append(hive_number)
            '''
            if data['params'] is True:
                print_debug("if data['params'] is True:")
                if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                    print_debug("if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                    for device_id, info in raspberrypi3B.devices_infos_dict.items():
                        print_debug("for device_id, info in raspberrypi3B.devices_infos_dict.items():")
                        location = info['location']
                        if hive_number == raspberrypi3B.get_hive_number_from_location(location):
                            print_debug("if hive_number == raspberrypi3B.get_hive_number_from_location(location):")
                            device_id_to_put_to_sleep = device_id
                            print(f"device_id_to_put_to_sleep: {device_id_to_put_to_sleep}, location: {location}")
                            # [{device_id1: hive_number1}, {device_id2: hive_number2}, ..., {device_idN: hive_numberN}]
                            raspberrypi3B.list_of_devices_hives_to_put_to_sleep.append({device_id_to_put_to_sleep : hive_number})
                            raspberrypi3B.delete_device_from_the_net(device_id_to_put_to_sleep)
                            break


                    print(f"raspberrypi3B.list_of_devices_hives_to_put_to_sleep: {raspberrypi3B.list_of_devices_hives_to_put_to_sleep}")
                    # Update the related led active status in thingsboard
                    thingsboard_leds = {}
                    list_of_devices = []
                    for device_id, info in raspberrypi3B.devices_infos_dict.items():
                        print_debug("for device_id, info in raspberrypi3B.devices_infos_dict.items():")
                        location = raspberrypi3B.devices_infos_dict[device_id]['location']
                        hive_number = raspberrypi3B.get_hive_number_from_location(location)
                        list_of_devices.append(hive_number)
                    for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):
                        print_debug("for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):")
                        key_led = "ledStatus" + str(i)
                        # if i not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep() and i in list_of_devices:
                        if i in list_of_devices:
                            thingsboard_leds.update({key_led: 1})
                        else:
                            thingsboard_leds.update({key_led: 0})
                    payload = json.dumps(thingsboard_leds)
                    # Publish data to the device telemetry topic
                    result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                    # mosquitto_pub -d -q 1 -h localhost -p 1883 -t v1/devices/me/telemetry -u "tGVWWbIrFr9PDDDxmUwJ" -m "{temperature:25}"
                    # Check if the publish was successful
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print("Data sent:", payload)
                    else:
                        print("Failed to send data. Return code:", result.rc)

                print("devices_to_make_an_audio_request_list:", raspberrypi3B.devices_to_make_an_audio_request_list)

            elif data['params'] is False:
                print_debug("elif data['params'] is False:")
                if hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                    print_debug("if hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                    raspberrypi3B.remove_from_list_of_devices_hives_to_put_to_sleep_given_the_hive_number(hive_number)
            print(f"list_of_devices_hives_to_put_to_sleep: {raspberrypi3B.list_of_devices_hives_to_put_to_sleep}")



        if data['method'].startswith("set_state_Telegram_switch"):
            print_debug("if data['method'].startswith('set_state_Telegram_switch'):")
            if data['params'] is True:
                print_debug("if data['params'] is True:")
                raspberrypi3B.telegram_notifications = True
            elif data['params'] is False:
                print_debug("if data['params'] is False:")
                raspberrypi3B.telegram_notifications = False

        if data['method'].startswith("set_time_min/"):
            print_debug("if data['method'].startswith('set_time_min/'):")
            splitted_topic = data['method'].split('/')
            minutes_between_measures = int(splitted_topic[-1])
            print(f"minutes_between_measures: {minutes_between_measures}")

            ten_min_in_sec = 600 # 600 normally, 90 for test
            thirty_min_in_sec = 1800 # 1800 normally, 180 for test
            one_hour_in_sec = 3600 # 3600 normally, 270 for test


            if minutes_between_measures == 10:
                print("if minutes_between_measures == 10:")
                raspberrypi3B.time_between_a_series_of_audio_requests = ten_min_in_sec # 90
                raspberrypi3B.esp32_deep_sleep_time = ten_min_in_sec-30 # 90 - 30
                led10min = True
                led30min = False
                led1hour = False
                if raspberrypi3B.current_timer >= ten_min_in_sec - 40:
                    increase_last_audio_request_publish_time = raspberrypi3B.current_timer - (ten_min_in_sec - 60)
                    raspberrypi3B.current_timer = ten_min_in_sec - 60
                    raspberrypi3B.last_audio_request_publish_time += increase_last_audio_request_publish_time
            elif minutes_between_measures == 30:
                print("if minutes_between_measures == 30:")
                raspberrypi3B.time_between_a_series_of_audio_requests = thirty_min_in_sec # 180
                raspberrypi3B.esp32_deep_sleep_time = thirty_min_in_sec - 30 # 180 - 30
                led10min = False
                led30min = True
                led1hour = False
                if raspberrypi3B.current_timer >= thirty_min_in_sec - 40:
                    increase_last_audio_request_publish_time = raspberrypi3B.current_timer - (thirty_min_in_sec - 60)
                    raspberrypi3B.current_timer = thirty_min_in_sec - 60
                    raspberrypi3B.last_audio_request_publish_time += increase_last_audio_request_publish_time
            elif minutes_between_measures == 60:
                print("if minutes_between_measures == 60:")
                raspberrypi3B.time_between_a_series_of_audio_requests = one_hour_in_sec # 270
                raspberrypi3B.esp32_deep_sleep_time = one_hour_in_sec - 30 # 270 - 30
                led10min = False
                led30min = False
                led1hour = True
                if raspberrypi3B.current_timer >= one_hour_in_sec - 40:
                    increase_last_audio_request_publish_time = raspberrypi3B.current_timer - (one_hour_in_sec - 60)
                    raspberrypi3B.current_timer = one_hour_in_sec - 60
                    raspberrypi3B.last_audio_request_publish_time += increase_last_audio_request_publish_time
            else: # default 10 min
                print("else")
                raspberrypi3B.time_between_a_series_of_audio_requests = ten_min_in_sec # 90
                led10min = True
                led30min = False
                led1hour = False
                if raspberrypi3B.current_timer >= ten_min_in_sec - 40:
                    increase_last_audio_request_publish_time = raspberrypi3B.current_timer - (ten_min_in_sec - 60)
                    raspberrypi3B.current_timer = ten_min_in_sec - 60
                    raspberrypi3B.last_audio_request_publish_time += increase_last_audio_request_publish_time
            print(f"raspberrypi3B.time_between_a_series_of_audio_requests: {raspberrypi3B.time_between_a_series_of_audio_requests}")
            payload = json.dumps({
                "led10min": led10min,
                "led30min": led30min,
                "led1hour": led1hour
            })
            if raspberrypi3B.caution_time_period_before_audio_request <= raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer < raspberrypi3B.caution_time_period_before_audio_request+1:
                print("if raspberrypi3B.caution_time_period_before_audio_request <= raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer < raspberrypi3B.caution_time_period_before_audio_request+1:")
                raspberrypi3B.imminent_audio_request_led_flag_thingsboard = True
            else:
                raspberrypi3B.imminent_audio_request_led_flag_thingsboard = False

            print(f"payload: {payload}")
            # Publish data to the device telemetry topic
            result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
            # Check if the publish was successful
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("Data sent:", payload)
            else:
                print("Failed to send data. Return code:", result.rc)









###################################################################################

'''
switch:                 sleep_Hive1, ..., sleep_Hive10
Device admin table:     active_devices
Alarms:                 Time, Hive, Status
Measure Period:         getMeasurePeriod, setMeasurePeriod --> time_between_a_series_of_audio_requests, max_time_without_communication_with_a_device
'''


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
    # def on_message(client, userdata, msg, raspberrypi3B):
    print_debug("on_message")
    # print_debug(f"Message received on topic {msg.topic}: {msg.payload.decode('utf-8')}")
    print_debug(f"Message received on topic '{msg.topic}'")
    if msg.topic.startswith("audio/payload") is False:
        print(msg.payload)
    print(f"raspberrypi3B.subscription_in_progress: {raspberrypi3B.subscription_in_progress}")

    # Initialize a list to hold audio chunks
    audio_chunks = []

    if msg.topic == "Raspberry_audio_available":
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("if msg.topic == Raspberry_audio_available:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        # 2      ---------------------------------------------------->{communication_id:<ID>,infos:{infos}}

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

            #decrypted_message_json = "{" + decrypted_message
            #decrypted_message_json = clean_json_string(decrypted_message_json)

            if '"simulation": true' in decrypted_message:
                print("simulation")
                decrypted_message_json = decrypted_message
            else:
                decrypted_message_json = "{" + decrypted_message
                decrypted_message_json = clean_json_string(decrypted_message_json)
                print("Decrypted Message JSON:", decrypted_message_json)

            if is_valid_json(decrypted_message_json):
                print("Decrypted Message JSON:", decrypted_message_json)
                json_received = json.loads(decrypted_message_json)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################


        try:
            print(f"json_received['infos']['device_id']: {json_received['infos']['device_id']}")
            print(f"raspberrypi3B.devices_infos_dict: {raspberrypi3B.devices_infos_dict}")
            if json_received["infos"]["device_id"] in raspberrypi3B.devices_infos_dict:
                print("if json_received['infos']['device_id'] in raspberrypi3B.devices_infos_dict:")
                device_id = json_received["infos"]["device_id"]
                print(f"BEFORE APPENDING TO devices_to_make_an_audio_request_list-->devices_to_make_an_audio_request_list: {raspberrypi3B.devices_to_make_an_audio_request_list}")
                raspberrypi3B.append_device_with_ready_audio_to_send_list(device_id)
                print(f"AFTER APPENDING TO devices_to_make_an_audio_request_list-->devices_to_make_an_audio_request_list: {raspberrypi3B.devices_to_make_an_audio_request_list}")
                raspberrypi3B.update_last_communication_epoch_dict(device_id)
                raspberrypi3B.update_devices_infos_dict(json_received["infos"])  # updating the infos of the device (where key=json_received["infos"]["device_id"] and value=json_received["infos"] without the device_id )
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("RaspberryPi: problem receiving the package for the topic 'Raspberry_get_infos'")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    elif msg.topic == "Raspberry_connect_to_sensors_net/ack":
    # elif msg.topic == "Raspberry_connect_to_sensors_net/ack" and raspberrypi3B.subscription_in_progress is True:
        #elif msg.topic == "Raspberry_connect_to_sensors_net/ack":
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("if msg.topic == Raspberry_connect_to_sensors_net/ack:")
        # print("if msg.topic == Raspberry_connect_to_sensors_net/ack and raspberrypi3B.subscription_in_progress is True:")
        #print("if msg.topic == Raspberry_connect_to_sensors_net/ack:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}, "net_password":<password>}}  (infos with device_id=None)
        # 2 {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>}<----------------------------------------------
        # 3 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}}}  (infos with device_id different than None)

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

            # decrypted_message_json = "{" + decrypted_message
            # decrypted_message_json = clean_json_string(decrypted_message_json)

            if '"simulation": true' in decrypted_message:
                print("simulation")
                decrypted_message_json = decrypted_message
            else:
                decrypted_message_json = "{" + decrypted_message
                decrypted_message_json = clean_json_string(decrypted_message_json)
                print("Decrypted Message JSON:", decrypted_message_json)

            if is_valid_json(decrypted_message_json):
                print("Decrypted Message JSON:", decrypted_message_json)
                json_received = json.loads(decrypted_message_json)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            #json_received_invitation_id = json_received["invitation_id"]
            #print(f"raspberrypi3B.current_invitation_id: {raspberrypi3B.current_invitation_id}, while json_received['invitation_id']: {json_received_invitation_id}")
            if json_received["invitation_id"] == raspberrypi3B.current_invitation_id:  # then we are in the case below
                print_debug("if json_received['invitation_id'] == raspberrypi3B.current_invitation_id:")
                # 3 ------------------------------->{invitation_id: < ID >, infos: {infos}}}  (infos with device_id different than None)
                device_id = json_received["infos"]["device_id"]
                if raspberrypi3B.subscription_in_progress is True:
                    print("if raspberrypi3B.subscription_in_progress is True:")
                    raspberrypi3B.update_last_communication_epoch_dict(device_id)
                    raspberrypi3B.update_devices_infos_dict(json_received["infos"])  # adding the infos of the new ESP32 added
                    print_debug(f"raspberrypi3B.devices_infos_dict: {raspberrypi3B.devices_infos_dict}")
                    raspberrypi3B.num_current_devices += 1  # updating the number of devices currently in the sensors network
                    raspberrypi3B.subscription_in_progress = False  # from now on, other devices can ask for a subscription to be part of the sensors network
                    raspberrypi3B.update_thingsboard_led_status = True
                else:
                    print("elif raspberrypi3B.subscription_in_progress is False:")
                    raspberrypi3B.update_last_communication_epoch_dict(device_id)
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug(
                "RaspberryPi: problem receiving the package for the topic 'Raspberry_connect_to_sensors_net/ack'")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")











    elif msg.topic == "Raspberry_connect_to_sensors_net/request" and raspberrypi3B.subscription_in_progress is False:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("elif msg.topic == Raspberry_connect_to_sensors_net/request and raspberrypi3B.subscription_in_progress is False:")
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

            # decrypted_message_json = "{" + decrypted_message
            # decrypted_message_json = clean_json_string(decrypted_message_json)

            if '"simulation": true' in decrypted_message:
                print("simulation")
                decrypted_message_json = decrypted_message
            else:
                decrypted_message_json = "{" + decrypted_message
                decrypted_message_json = clean_json_string(decrypted_message_json)
                print("Decrypted Message JSON:", decrypted_message_json)

            if is_valid_json(decrypted_message_json):
                print("Valid Decrypted Message JSON:", decrypted_message_json)
                json_received = json.loads(decrypted_message_json)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################


        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}, "net_password":<password>}}  (infos with device_id=None)
        # 2 {invitation_id:<ID>, device_id:<ID>, Connection_permission:<allowed/denied>}<----------------------------------------------
        # 3 -------------------------------------------------------------------------------------------->{invitation_id: <ID>, infos: {infos}}}  (infos with device_id different than None)
        try:
            print_debug(f"json_received: {json_received}")
            location = json_received['infos']['location']
            hive_number = raspberrypi3B.get_hive_number_from_location(location)
            print(f"hive_number: {hive_number}")
            print(f"raspberrypi3B.num_current_devices: {raspberrypi3B.num_current_devices}")
            print(f"raspberrypi3B.num_max_allowed_devices: {raspberrypi3B.num_max_allowed_devices}")
            if raspberrypi3B.num_current_devices < raspberrypi3B.num_max_allowed_devices:  # the device's request to be part of the sensors network will be accepted
                print_debug("if raspberrypi3B.num_current_devices < raspberrypi3B.num_max_allowed_devices:")
                # print_debug(f"json_received['infos']['device_id'] == '': {json_received['infos']['device_id'] == ''}")
                # print_debug(f"'net_password' in json_received: {'net_password' in json_received}")
                # print_debug(f"json_received['infos']['device_id'] == '' and 'net_password' in json_received:{json_received['infos']['device_id'] == '' and 'net_password' in json_received}")
                if json_received["net_password"] == raspberrypi3B.net_password:
                    print_debug("if json_received['net_password'] == raspberrypi3B.net_password:")
                    print(f"raspberrypi3B.devices_fixed_id_dict: {raspberrypi3B.devices_fixed_id_dict}")
                    # 1 ------------------------------->{invitation_id: <ID>, infos: {infos}, "net_password":<password>}}  infos with device_id=None)
                    # the device is already part of the sensors network
                    if json_received['infos']['fixed_id'] in raspberrypi3B.devices_fixed_id_dict:
                        print("if json_received['infos']['fixed_id'] in raspberrypi3B.devices_fixed_id_dict:")
                        if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                            print("if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                            fixed_id = json_received['infos']['fixed_id']
                            device_id = raspberrypi3B.devices_fixed_id_dict[fixed_id]
                            publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack",
                                                                     json_received["invitation_id"],
                                                                     "already_in",
                                                                     device_id, raspberrypi3B)
                            raspberrypi3B.update_last_communication_epoch_dict(device_id)
                        elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                            print_debug("elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                            publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack",
                                                                     json_received["invitation_id"], "denied", None,
                                                                     raspberrypi3B)
                    elif json_received['infos']['fixed_id'] not in raspberrypi3B.devices_fixed_id_dict:  # the device is not part of the sensors network, and from now on it will be part of it
                        print("the device is not part of the sensors network, and from now on it will be part of it")
                        if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                            print("if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                            #raspberrypi3B.subscription_in_progress = True  # till the flag is True, no other devices can ask for a subscription to be part of the sensors network
                            #raspberrypi3B.subscription_in_progress_timer = time.time()
                            raspberrypi3B.current_invitation_id = json_received["invitation_id"]
                            assigned_device_id = generate_random_code(20)
                            #time.sleep(1)
                            raspberrypi3B.devices_fixed_id_dict.update({json_received['infos']['fixed_id']: assigned_device_id})
                            publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack",
                                                                     json_received["invitation_id"],
                                                                     "allowed",
                                                                     assigned_device_id, raspberrypi3B)
                            raspberrypi3B.subscription_in_progress = True  # till the flag is True, no other devices can ask for a subscription to be part of the sensors network
                            raspberrypi3B.subscription_in_progress_timer = time.time()
                            raspberrypi3B.update_last_communication_epoch_dict(assigned_device_id)
                        elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                            print_debug("elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                            publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack",
                                                                     json_received["invitation_id"], "denied", None, raspberrypi3B)
            elif raspberrypi3B.num_current_devices >= raspberrypi3B.num_max_allowed_devices:  # the device's request to be part of the sensors network will not be accepted since there is no space
                print_debug("elif raspberrypi3B.num_current_devices >= raspberrypi3B.num_max_allowed_devices:")
                if json_received["net_password"] == raspberrypi3B.net_password and json_received['infos']['fixed_id'] in raspberrypi3B.devices_fixed_id_dict:
                    print_debug("if json_received['net_password'] == raspberrypi3B.net_password and json_received['infos']['fixed_id'] in raspberrypi3B.devices_fixed_id_dict:")
                    print(f"raspberrypi3B.devices_fixed_id_dict: {raspberrypi3B.devices_fixed_id_dict}")
                    # 1 ------------------------------->{invitation_id: <ID>, infos: {infos}, "net_password":<password>}}  infos with device_id=None)
                    # the device is already part of the sensors network
                    if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                        print("if hive_number not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                        fixed_id = json_received['infos']['fixed_id']
                        device_id = raspberrypi3B.devices_fixed_id_dict[fixed_id]
                        publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack", json_received["invitation_id"],"already_in", device_id, raspberrypi3B)
                        raspberrypi3B.update_last_communication_epoch_dict(device_id)
                    elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                        print_debug("elif hive_number in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():")
                        publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack", json_received["invitation_id"], "denied", None, raspberrypi3B)
                else:
                    print("else:")
                    publish_response_to_subscription_request(client, "NodeMCU_connect_to_sensors_net/ack", json_received["invitation_id"], "denied", None, raspberrypi3B)

        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug(
                "RaspberryPi: problem receiving the package for the topic 'Raspberry_connect_to_sensors_net/request'")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")



    elif msg.topic == "Raspberry_get_infos":
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("if msg.topic == Raspberry_get_infos:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1      {device_id:<ID>, communication_id:<ID>}<--------------------------------------------------
        # 2      ---------------------------------------------------->{communication_id:<ID>,infos:{infos}}

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

            # decrypted_message_json = "{" + decrypted_message
            # decrypted_message_json = clean_json_string(decrypted_message_json)

            if '"simulation": true' in decrypted_message:
                print("simulation")
                decrypted_message_json = decrypted_message
            else:
                decrypted_message_json = "{" + decrypted_message
                decrypted_message_json = clean_json_string(decrypted_message_json)
                print("Decrypted Message JSON:", decrypted_message_json)

            if is_valid_json(decrypted_message_json):
                print("Decrypted Message JSON:", decrypted_message_json)
                json_received = json.loads(decrypted_message_json)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################
        try:
            if json_received["communication_id"] == raspberrypi3B.communication_id:
                raspberrypi3B.dev_is_connected = True
                device_id = json_received["infos"]["device_id"]
                raspberrypi3B.update_last_communication_epoch_dict(device_id)
                raspberrypi3B.update_devices_infos_dict(json_received["infos"])  # updating the infos of the device (where key=json_received["infos"]["device_id"] and value=json_received["infos"] without the device_id )
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("RaspberryPi: problem receiving the package for the topic 'Raspberry_get_infos'")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    elif raspberrypi3B.audio_data_topic_key is not None:
        if msg.topic == "audio/payload" + raspberrypi3B.audio_data_topic_key:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(f"msg.topic={msg.topic}")
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            print_debug(f"formatted_time: {formatted_time}")
            try:
                save_audio_from_payload(client, msg, raspberrypi3B)
            except Exception as e:
                print(f"Failed to save the audio: {str(e)}")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    elif msg.topic == "Raspberry_recording_problems":
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("if msg.topic == Raspberry_recording_problems:")
        local_time = time.localtime(time.time())
        # Format struct_time to a string
        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        print_debug(f"formatted_time: {formatted_time}")
        # 1      ------------------------------------>{communication_id:<ID>,infos:{infos}}
        # 2      {communication_id:<ID>}<--------------------------------------------------

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

            # decrypted_message_json = "{" + decrypted_message
            # decrypted_message_json = clean_json_string(decrypted_message_json)

            if '"simulation": true' in decrypted_message:
                print("simulation")
                decrypted_message_json = decrypted_message
            else:
                decrypted_message_json = "{" + decrypted_message
                decrypted_message_json = clean_json_string(decrypted_message_json)
                print("Decrypted Message JSON:", decrypted_message_json)

            if is_valid_json(decrypted_message_json):
                print("Decrypted Message JSON:", decrypted_message_json)
                json_received = json.loads(decrypted_message_json)
            else:
                print("The message received is not a JSON")
                decrypted_message_string = decrypted_message.split('-')[0] + '-'
                print("Decrypted Message String:", decrypted_message_string)

        except Exception as e:
            print(f"Failed to process message: {str(e)}")
        print("________________________")

        # END OF DECRYPTION ##############################################################

        try:
            if json_received["infos"]["device_id"] in raspberrypi3B.devices_infos_dict:
                raspberrypi3B.update_devices_infos_dict(json_received["infos"])  # updating the infos of the device (where key=json_received["infos"]["device_id"] and value=json_received["infos"] without the device_id)
                communication_id = json_received["communication_id"]
                device_id = json_received["infos"]["device_id"]
                raspberrypi3B.update_last_communication_epoch_dict(device_id)
                message_to_send_back = {"communication_id": communication_id}
                message_to_send_back = json.dumps(message_to_send_back)
                local_time = time.localtime(time.time())
                # Format struct_time to a string
                formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
                print_debug("+++++++++++++++++++++++++++++")
                print_debug(f"formatted_time: {formatted_time}")
                print_debug(f"topic: NodeMCU_recording_problems")
                encrypted_message_to_send_back = encrypt(message_to_send_back)
                client.publish("NodeMCU_recording_problems", encrypted_message_to_send_back, qos=2) # client
            else:  # the device_id is unknown, thus for security problems, it is better to not resend back any message
                pass
        except json.JSONDecodeError:
            print_debug("Received payload is not valid JSON.")
            print_debug("RaspberryPi: problem receiving the package for the topic 'Raspberry_recording_problems'")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("EXIT FROM on_message__________________________________________________________________________")


def run_cpp_executable_for_inference_classification():
    print_debug("run_cpp_executable_for_inference_classification")
    # Get the current time in microseconds
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    microseconds = int((time.time() % 1) * 1_000_000)
    # Combine formatted time with microseconds
    formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
    print_debug(f"run_cpp_executable1 at {formatted_time_with_micros}")

    # Here, 'your_cpp_program' is the name/path of your C++ executable
    # command example:     ./inference_RPW_spectrogram_binaries2_exe TreeVibes_model_batch8_new.tflite /home/pi/aarch64_build_ref-neon/python_env/RPW_audio_preprocessing/received_audio_files CpuAcc >> audio_files_collector/inference_results.txt

    # Command as a list
    command = [
        './inference_QueenBees_spectrogram_binaries_exe',
        'queen_bee_presence_prediction.tflite',
        '/home/pi/aarch64_build_ref-neon/received_audio_files',
        'CpuAcc'
    ]

    try:
        # Open the output file in append mode
        with open('audio_files_collector/inference_results.txt', 'a') as output_file:  # Change 'w' to 'a'
            # Start the subprocess
            process = subprocess.Popen(command, stdout=output_file, stderr=subprocess.STDOUT)
            process.wait()  # Wait for the process to complete

        # Check if the process exited successfully
        if process.returncode == 0:
            print("Command executed successfully!")
        else:
            print(f"Command failed with return code {process.returncode}")

    except FileNotFoundError:
        print_debug("C++ program not found. Make sure the executable path is correct.")
    except Exception as e:
        print_debug(f"An error occurred while running the C++ program: {str(e)}")


##############################################################

SR = 22050
N_FFT = 1024
HOP_LEN = int(N_FFT / 2)
n_chunks = 27
input_shape = (27, 44, 1)


##############################################################

def parse_config_file(file_path):
    print_debug("parse_config_file")
    audio_folder = None
    save_folder = None
    tflite_model_path = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('#audio_folder'):
                audio_folder = next(file).strip()  # Read next line for the path
            elif line.startswith('#save_folder'):
                save_folder = next(file).strip()  # Read next line for the path
            elif line.startswith('#tflite_model_path'):
                tflite_model_path = next(file).strip()  # Read next line for the path

    return audio_folder, save_folder, tflite_model_path


# ---------mean summarization function------#
def mean(s, n_chunks):
    m, f = s.shape
    mod = m % n_chunks
    # print(mod)
    if m % n_chunks != 0:
        s = np.delete(s, np.s_[0:mod], 0)
    stft_mean = []
    split = np.split(s, n_chunks, axis=0)
    for i in range(0, n_chunks):
        stft_mean.append(split[i].mean(axis=0))
    stft_mean = np.asarray(stft_mean)
    return stft_mean
# ------------------------------------------#

# --------feature extraction tools----------#
# stft
def stft_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                            center=True, dtype=np.complex64, pad_mode='reflect'))
    # m, t, s = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False,
    # return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    summ_s = mean(s, n_chunks)
    return summ_s
# ------------------------------------------#

# Function to preprocess audio files
def preprocess_audio(filepath, n_chunks):
    input_feature = stft_extraction(filepath, n_chunks)
    input_feature = np.expand_dims(input_feature, axis=-1)
    return input_feature

def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1(audio_folder, file_list):
    print_debug("preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1")

    print_debug(">>>>>>>>>>>time when entering preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1")
    # Get the current time in microseconds
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    microseconds = int((time.time() % 1) * 1_000_000)
    # Combine formatted time with microseconds
    formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
    print_debug(f"{formatted_time_with_micros}")

    # Classify each audio file in the folder
    # audio_folder = "received_audio_files/thread1/"
    # audio_folder = "received_audio_files/"
    from pathlib import Path
    # for root, _, files in os.walk(audio_folder):
    # Get the root directory from the audio_folder
    root = Path(audio_folder).resolve()
    print_debug(f"file_list: {file_list}")
    print_debug(f"Root Directory: {root}")
    for file in file_list:
        if file.endswith(".wav"):
            print(f"file: {file}")
            output_file_name = str(Path(file).stem) + '.bin'
            input_file_name = str(Path(file).stem) + '.wav'
            save_folder = audio_folder
            processed_audio = preprocess_audio(os.path.join(root, input_file_name), n_chunks)
            output_file_name = output_file_name
            with open(save_folder + output_file_name, 'wb') as f:
                processed_audio.tofile(f)  # Write the raw byte data to the binary file
            print(f"Saved {output_file_name}")
            print_debug("time when exiting preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1")
            # Get the current time in microseconds
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            microseconds = int((time.time() % 1) * 1_000_000)
            # Combine formatted time with microseconds
            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
            print_debug(f"{formatted_time_with_micros}")
            print_debug("___________________________")


def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2(audio_folder, file_list):
    print_debug("preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2")

    print_debug(">>>>>>>>>>>time when entering preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2")
    # Get the current time in microseconds
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    microseconds = int((time.time() % 1) * 1_000_000)
    # Combine formatted time with microseconds
    formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
    print_debug(f"{formatted_time_with_micros}")

    # Classify each audio file in the folder
    # audio_folder = "received_audio_files/thread1/"
    # audio_folder = "received_audio_files/"
    from pathlib import Path
    # for root, _, files in os.walk(audio_folder):
    # Get the root directory from the audio_folder
    root = Path(audio_folder).resolve()
    print_debug(f"file_list: {file_list}")
    print_debug(f"Root Directory: {root}")
    for file in file_list:
        if file.endswith(".wav"):
            print(f"file: {file}")
            output_file_name = str(Path(file).stem) + '.bin'
            input_file_name = str(Path(file).stem) + '.wav'
            save_folder = audio_folder
            processed_audio = preprocess_audio(os.path.join(root, input_file_name), n_chunks)
            output_file_name = output_file_name
            with open(save_folder + output_file_name, 'wb') as f:
                processed_audio.tofile(f)  # Write the raw byte data to the binary file
            print(f"Saved {output_file_name}")
            print_debug("time when exiting preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2")
            # Get the current time in microseconds
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            microseconds = int((time.time() % 1) * 1_000_000)
            # Combine formatted time with microseconds
            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
            print_debug(f"{formatted_time_with_micros}")
            print_debug("___________________________")


def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3(audio_folder, file_list):
    print_debug("preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3")

    print_debug(">>>>>>>>>>>time when entering preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3")
    # Get the current time in microseconds
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    microseconds = int((time.time() % 1) * 1_000_000)
    # Combine formatted time with microseconds
    formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
    print_debug(f"{formatted_time_with_micros}")

    # Classify each audio file in the folder
    # audio_folder = "received_audio_files/thread1/"
    # audio_folder = "received_audio_files/"
    from pathlib import Path
    # for root, _, files in os.walk(audio_folder):
    # Get the root directory from the audio_folder
    root = Path(audio_folder).resolve()
    print_debug(f"file_list: {file_list}")
    print_debug(f"Root Directory: {root}")
    for file in file_list:
        if file.endswith(".wav"):
            print(f"file: {file}")
            output_file_name = str(Path(file).stem) + '.bin'
            input_file_name = str(Path(file).stem) + '.wav'
            save_folder = audio_folder
            processed_audio = preprocess_audio(os.path.join(root, input_file_name), n_chunks)
            output_file_name = output_file_name
            with open(save_folder + output_file_name, 'wb') as f:
                processed_audio.tofile(f)  # Write the raw byte data to the binary file
            print(f"Saved {output_file_name}")
            print_debug("time when exiting preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3")
            # Get the current time in microseconds
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            microseconds = int((time.time() % 1) * 1_000_000)
            # Combine formatted time with microseconds
            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
            print_debug(f"{formatted_time_with_micros}")
            print_debug("___________________________")


def preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4(audio_folder, file_list):
    print_debug("preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4")

    print_debug(">>>>>>>>>>>time when entering preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4")
    # Get the current time in microseconds
    local_time = time.localtime(time.time())
    # Format struct_time to a string
    formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
    microseconds = int((time.time() % 1) * 1_000_000)
    # Combine formatted time with microseconds
    formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
    print_debug(f"{formatted_time_with_micros}")

    # Classify each audio file in the folder
    # audio_folder = "received_audio_files/thread1/"
    # audio_folder = "received_audio_files/"
    from pathlib import Path
    # for root, _, files in os.walk(audio_folder):
    # Get the root directory from the audio_folder
    root = Path(audio_folder).resolve()
    print_debug(f"file_list: {file_list}")
    print_debug(f"Root Directory: {root}")
    for file in file_list:
        if file.endswith(".wav"):
            print(f"file: {file}")
            output_file_name = str(Path(file).stem) + '.bin'
            input_file_name = str(Path(file).stem) + '.wav'
            save_folder = audio_folder
            processed_audio = preprocess_audio(os.path.join(root, input_file_name), n_chunks)
            output_file_name = output_file_name
            with open(save_folder + output_file_name, 'wb') as f:
                processed_audio.tofile(f)  # Write the raw byte data to the binary file
            print(f"Saved {output_file_name}")
            print_debug("time when exiting preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4")
            # Get the current time in microseconds
            local_time = time.localtime(time.time())
            # Format struct_time to a string
            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
            microseconds = int((time.time() % 1) * 1_000_000)
            # Combine formatted time with microseconds
            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
            print_debug(f"{formatted_time_with_micros}")
            print_debug("___________________________")


def count_wav_files_and_list_them(directory, audio_folder):
    print_debug("count_wav_files_and_list_them")
    try:
        # List all files in the specified directory
        files = os.listdir(directory)

        # Count .wav files
        wav_file_count = sum(1 for file in files if file.endswith('.wav'))

        # Collect all files in the directory
        all_files = []

        for root, _, files in os.walk(audio_folder):
            # Store files in the order they appear
            all_files.extend(os.path.join(root, file) for file in files)

        return wav_file_count, all_files
    except Exception as e:
        print_debug(f"An error occurred: {e}")
        return 0, []  # Return 0 if an error occurs, and an empty list


def move_files(source_folder, destination_folder):
    # Make sure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        # Check if it's a file
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_folder)
    print_debug("All files have been moved.")



def split_audio_into_segments(input_folder, output_folder, filenames, segment_duration_sec=1):
    """
    Splits audio files into segments of a specified duration, discarding segments shorter than the specified duration.

    Args:
        input_folder (str): The path to the folder containing the audio files.
        output_folder (str): The path to the folder where the segmented audio files will be saved.
        segment_duration_sec (float, optional): The duration of each segment in seconds. Defaults to 1 sec.
    """

    for filename in filenames:
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)

            try:
                # Load audio
                y, sr = librosa.load(input_path, sr=None)

                segment_samples = int(segment_duration_sec * sr)
                segment_counter = 0 # only 3 segments from a file
                for i in range(0, len(y), segment_samples):
                    segment = y[i:i + segment_samples]

                    # Skip segments shorter than the specified duration
                    if len(segment) < segment_samples:
                        if len(segment) > 0: # If there is something on it, continue to next iteration
                            print(f"Skipping short segment of {filename}")
                            continue
                        else:
                            break # exit for loop

                    # Normalize the audio
                    segment_normalized = librosa.util.normalize(segment)

                    # Convert to 16-bit audio
                    segment_int16 = (segment_normalized * 32767).astype(np.int16)

                    # Create output filename
                    output_filename = f"{os.path.splitext(filename)[0]}_segment_{i // segment_samples:03d}.wav"
                    output_path = os.path.join(output_folder, output_filename)

                    # Save the segment
                    sf.write(output_path, segment_int16, sr, subtype='PCM_16')
                    print(f"Created segment {i // segment_samples + 1} of {filename} and saved to {output_filename}")
                    segment_counter += 1
                    if segment_counter >= 3: break # only 3 segments from a file
                # deleting the original file (thus leaving only the segmented files created from it)
                os.remove(input_path)
                print(f"Deleted the file: {input_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Check if the location's substring obtained from a .wav filename matches any location in the devices_infos_dict, and it returns the related device_id
def get_device_id_from_location(location_substring, RaspberryPi):
    try:
        for device_id, device_info in RaspberryPi.devices_infos_dict.items():
            if device_info['location'] == location_substring:
                return device_id
    except Exception as e:
        print(f"Error in check_location_exists function: {e}")



def classify_audios_by_labels_and_update_state(original_filenames, folder_with_segment_files, RaspberryPi):
    names_of_segment_files = os.listdir(folder_with_segment_files)
    # Filter the filenames that contain any of the specified substrings

    try:
        for filename in original_filenames:
            root_original_filenames, ext = os.path.splitext(filename)
            substrings = [root_original_filenames, 'noqueen', '.wav']  # List of substrings to coexist together to search for in names_of_segment_files
            filtered_names_of_segment_files = [filename for filename in names_of_segment_files if all(substring in filename for substring in substrings)]
            print(f"filtered_names_of_segment_files: {filtered_names_of_segment_files}")
            location_list = re.findall(r'\((.*?)\)',root_original_filenames)  # extracts from the filename the location that identifies the current beehive
            print(f"location: {location_list[0]}")
            device_id = get_device_id_from_location(location_list[0], RaspberryPi)
            print(f"device_id: {device_id}")
            if raspberrypi3B.split_3sec_audio_flag is True:
                print("if raspberrypi3B.split_3sec_audio_flag is True:")
                if len(filtered_names_of_segment_files) == 3: # 3/3 files are 'noqueen', 0/3 files are 'queen'
                    # substring_to_use = 'noqueen' # no need to rename the files
                    RaspberryPi.update_beehive_state(device_id, "noqueen")
                elif len(filtered_names_of_segment_files) == 2: # 2/3 files are 'noqueen', 1/3 files are 'queen'
                    substring_to_use = 'noqueen'
                    substrings_to_search = [root_original_filenames, 'queen']
                    file_to_rename = [filename for filename in names_of_segment_files if any('queen' in filename for substring in substrings_to_search)] # a list with just one file name
                    os.path.basename(file_to_rename[0]).replace('queen', substring_to_use)
                    RaspberryPi.update_beehive_state(device_id, "queen")
                elif len(filtered_names_of_segment_files) == 1: # 2/3 files are 'queen', 1/3 files are 'noqueen'
                    substring_to_use = 'queen'
                    os.path.basename(substrings[0]).replace('noqueen', substring_to_use)
                    RaspberryPi.update_beehive_state(device_id, "queen")
                elif len(filtered_names_of_segment_files) == 0: # 3/3 files are 'queen', 0/3 files are 'noqueen'
                    # substring_to_use = 'queen' # no need to rename the files
                    RaspberryPi.update_beehive_state(device_id, "queen")
                    pass
                else:
                    print("Error during classification: the number of segments of a file must be not more than 3!")
                print(f"- - - - - - infos of the classified beehive of device_id={device_id}: {RaspberryPi.devices_infos_dict[device_id]}")
            elif raspberrypi3B.split_3sec_audio_flag is False:
                print("elif raspberrypi3B.split_3sec_audio_flag is False:")
                if len(filtered_names_of_segment_files) == 1: # noqueen
                    RaspberryPi.update_beehive_state(device_id, "noqueen")
                elif len(filtered_names_of_segment_files) == 0: # queen
                    RaspberryPi.update_beehive_state(device_id, "queen")
                else:
                    print("Error during classification: more than one file with extension '.wav' have the same root_name and classification label!")

                print(f"- - - - - - infos of the classified beehive of device_id={device_id}: {RaspberryPi.devices_infos_dict[device_id]}" )

    except Exception as e:
        print(f"Error renaming files: {e}")



audio_files = []
target_names = ['clean', 'infested']  # Define your target class names


def main():
    mqtt_client_id = generate_random_code(12)  # all clients must have a different client_id

    # Create an MQTT client instance
    # client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
    client = mqtt.Client(client_id=mqtt_client_id, protocol=mqtt.MQTTv5)
    # client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    # Attach the callback functions
    print_debug("# Attach the callback functions")
    client.on_log = on_log
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    #THINGSBOARD###################################################################################################

    mqtt_thingsboard_client_id = generate_random_code(12)  # all clients must have a different client_id
    # Create an MQTT client
    client_thingsboard = mqtt.Client(client_id=mqtt_thingsboard_client_id, protocol=mqtt.MQTTv5)

    # Connect to the ThingsBoard MQTT broker. Use the access token as the username.
    client_thingsboard.username_pw_set(ACCESS_TOKEN, password=None)  # Password is not required

    client_thingsboard.on_log = on_log_thingsboard
    client_thingsboard.on_connect = on_connect_thingsboard
    client_thingsboard.on_disconnect = on_disconnect_thingsboard
    client_thingsboard.on_message = on_message_thingsboard
    ###############################################################################################################



    try:

        # Now connect to the host
        print_debug(f"Trying to connect with the broker {THINGSBOARD_HOST}")
        client_thingsboard.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, THINGSBOARD_KEEPALIVE)
        print_debug(f"Successfully connected to the broker {THINGSBOARD_HOST}")
        print_debug("client_thingsboard.loop_start()")
        client_thingsboard.loop_start()

        time.sleep(4)

        # Now connect to the host
        print_debug(f"Trying to connect with the broker {broker}")
        client.connect(broker, port, keepalive)
        print_debug(f"Successfully connected to the broker {broker}")
        # Start the loop in a separate thread to process network traffic and dispatch callbacks
        print_debug("client.loop_start()")
        client.loop_start()



        time.sleep(4)
        raspberrypi3B.last_audio_request_publish_time = time.time()

        audio_folder = "received_audio_files/"
        collector_folder = "audio_files_collector/"


        iter_dashboard = 0
        upload_dashboard_period = 15  # every 30 sec we upload the dashboard, except the 30 sec before the audio request
        raspberrypi3B.imminent_audio_request_led_flag_thingsboard = False

        # except Exception as e:
        #    print_debug(f"Could not connect to broker: {e}")
        #    logging.error(f"Exception occurred: {e}")
        #    exit()

        try:

            while True:

                current_time = time.time()

                # print_debug(f"raspberrypi3B.time_between_a_series_of_audio_requests: {raspberrypi3B.time_between_a_series_of_audio_requests}")
                print(f"raspberrypi3B.subscription_in_progress: {raspberrypi3B.subscription_in_progress}")
                print_debug(f"raspberrypi3B.current_timer = {raspberrypi3B.current_timer:.2f}")
                print_debug(f"iter_dashboard = {iter_dashboard:.2f}")
                raspberrypi3B.current_timer = current_time - raspberrypi3B.last_audio_request_publish_time
                print("raspberrypi3B.current_timer = current_time - raspberrypi3B.last_audio_request_publish_time")
                print(f"{raspberrypi3B.current_timer} = {current_time} - {raspberrypi3B.last_audio_request_publish_time}")

                if raspberrypi3B.caution_time_period_before_audio_request <= raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer < raspberrypi3B.caution_time_period_before_audio_request+1:
                    print("if raspberrypi3B.caution_time_period_before_audio_request <= raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer < raspberrypi3B.caution_time_period_before_audio_request+1:")
                    raspberrypi3B.imminent_audio_request_led_flag_thingsboard = True
                print(f"imminent_audio_request_led_flag: {raspberrypi3B.imminent_audio_request_led_flag_thingsboard}")

                # print_debug(f"len(raspberrypi3B.devices_to_make_an_audio_request_list): {len(raspberrypi3B.devices_to_make_an_audio_request_list)}")
                # print_debug(f"raspberrypi3B.num_current_devices: {raspberrypi3B.num_current_devices}")
                # print_debug(f"raspberrypi3B.counter_resend_request_done: {raspberrypi3B.counter_resend_request_done}")
                # print_debug(f"raspberrypi3B.subscription_in_progress: {raspberrypi3B.subscription_in_progress}")



                # it could happen that an already started subscription attempt doesn't come to an end... So the flag subscription_in_progress could stay True,
                # leading to the inability for other devices to subscribe (since they could subscribe only if subscription_in_progress == False). To solve this problem, subscription_in_progress_timer will be 0
                # whenever a subscription attempt starts, and after 15 seconds it will be put automatically to subscription_in_progress = False, in order for eventual other devices to subscribe successfully

                '''
                if raspberrypi3B.subscription_in_progress is True:
                    print("if raspberrypi3B.subscription_in_progress is True:")
                    print(f"time.time() - raspberrypi3B.subscription_in_progress_timer: {time.time() - raspberrypi3B.subscription_in_progress_timer}")
                    
                    if time.time() - raspberrypi3B.subscription_in_progress_timer >= 15:
                        print("if time.time() - raspberrypi3B.subscription_in_progress_timer >= 15:")
                        raspberrypi3B.subscription_in_progress = False


                    print(f"subscription_in_progress: {raspberrypi3B.subscription_in_progress}")
                    print(f"subscription_in_progress_timer: {raspberrypi3B.subscription_in_progress_timer}")
                '''


                # THINGSBOARD DASHBOARD UPDATE ####################################################################################
                if raspberrypi3B.update_thingsboard_led_status is True:
                    print_debug("if raspberrypi3B.update_thingsboard_led_status is True:")
                    raspberrypi3B.update_thingsboard_led_status = False

                    # Thingsboard Led indicators update ######################################################################
                    thingsboard_leds = {}
                    list_of_devices = []
                    for device_id, info in raspberrypi3B.devices_infos_dict.items():
                        location = raspberrypi3B.devices_infos_dict[device_id]['location']
                        hive_number = raspberrypi3B.get_hive_number_from_location(location)
                        list_of_devices.append(hive_number)

                    max_hive_number = raspberrypi3B.num_max_allowed_devices
                    if len(list_of_devices) > 0:
                        max_hive_number = max(list_of_devices)
                        if max_hive_number > raspberrypi3B.max_hive_number_ever_registered:
                            raspberrypi3B.max_hive_number_ever_registered = max_hive_number

                    for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):

                        # if i not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep() and i in list_of_devices:
                        if i in list_of_devices:
                            key_led = "ledStatus" + str(i)
                            thingsboard_leds.update({key_led: 1})
                        else:
                            key_led = "ledStatus" + str(i)
                            thingsboard_leds.update({key_led: 0})

                    payload = json.dumps(thingsboard_leds)
                    # Publish data to the device telemetry topic
                    result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                    # mosquitto_pub -d -q 1 -h localhost -p 1883 -t v1/devices/me/telemetry -u "tGVWWbIrFr9PDDDxmUwJ" -m "{temperature:25}"

                    # Check if the publish was successful
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print("Data sent:", payload)
                    else:
                        print("Failed to send data. Return code:", result.rc)



                # Avoid to update the thingsboard dashboard the 40 sec before the audio requests

                print(f"if {iter_dashboard} >= {upload_dashboard_period} and {raspberrypi3B.current_timer} < {(raspberrypi3B.time_between_a_series_of_audio_requests-raspberrypi3B.caution_time_period_before_audio_request)}:")
                if iter_dashboard >= upload_dashboard_period and raspberrypi3B.current_timer < (raspberrypi3B.time_between_a_series_of_audio_requests-raspberrypi3B.caution_time_period_before_audio_request) and raspberrypi3B.imminent_audio_request_led_flag_thingsboard is False:
                    print("ENTER: if iter_dashboard >= upload_dashboard_period and raspberrypi3B.current_timer < (raspberrypi3B.time_between_a_series_of_audio_requests-raspberrypi3B.caution_time_period_before_audio_request) and raspberrypi3B.imminent_audio_request_led_flag_thingsboard is False:")

                    if raspberrypi3B.subscription_in_progress is True:
                        print("if raspberrypi3B.subscription_in_progress is True:")
                        print(
                            f"time.time() - raspberrypi3B.subscription_in_progress_timer: {time.time() - raspberrypi3B.subscription_in_progress_timer}")

                        if time.time() - raspberrypi3B.subscription_in_progress_timer >= 15:
                            print("if time.time() - raspberrypi3B.subscription_in_progress_timer >= 15:")
                            raspberrypi3B.subscription_in_progress = False

                        print(f"subscription_in_progress: {raspberrypi3B.subscription_in_progress}")
                        print(f"subscription_in_progress_timer: {raspberrypi3B.subscription_in_progress_timer}")



                    # UPDATE IMMINENT AUDIO REQUEST LED
                    thingsboard_imminent_audio_request = {"imminent_audio_request_led": False}
                    # UPDATE TIME TO AUDIO REQUEST
                    time_to_audio_request = int(raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer)
                    thingsboard_time_to_audio_request = {"time_to_audio_request": time_to_audio_request}
                    # Thingsboard Led indicators update ######################################################################
                    thingsboard_leds = {}
                    list_of_devices = []
                    for device_id, info in raspberrypi3B.devices_infos_dict.items():
                        location = raspberrypi3B.devices_infos_dict[device_id]['location']
                        hive_number = raspberrypi3B.get_hive_number_from_location(location)
                        list_of_devices.append(hive_number)

                    max_hive_number = raspberrypi3B.num_max_allowed_devices
                    if len(list_of_devices) > 0:
                        max_hive_number = max(list_of_devices)
                        if max_hive_number > raspberrypi3B.max_hive_number_ever_registered:
                            raspberrypi3B.max_hive_number_ever_registered = max_hive_number

                    for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):
                        # if i not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep() and i in list_of_devices:
                        if i in list_of_devices:
                            key_led = "ledStatus" + str(i)
                            thingsboard_leds.update({key_led: 1})
                        else:
                            key_led = "ledStatus" + str(i)
                            thingsboard_leds.update({key_led: 0})

                    # Thingsboard Single switches update ######################################################################
                    thingsboard_switches = {}
                    print(raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep())

                    for i in range(1, max_hive_number):
                        if i in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep():
                            key_switch = "state_sleep_Hive/" + str(i)
                            thingsboard_switches.update({key_switch: True})
                        else:
                            key_switch = "state_sleep_Hive/" + str(i)
                            thingsboard_switches.update({key_switch: False})



                    # Thingsboard measure time led ######################################################################
                    print(
                        f"raspberrypi3B.time_between_a_series_of_audio_requests: {raspberrypi3B.time_between_a_series_of_audio_requests}")
                    if raspberrypi3B.time_between_a_series_of_audio_requests == 600:
                        thingsboard_led_measure_period = {
                            "led10min": True,
                            "led30min": False,
                            "led1hour": False
                        }
                    elif raspberrypi3B.time_between_a_series_of_audio_requests == 1800:
                        thingsboard_led_measure_period = {
                            "led10min": False,
                            "led30min": True,
                            "led1hour": False
                        }
                    elif raspberrypi3B.time_between_a_series_of_audio_requests == 3600:
                        thingsboard_led_measure_period = {
                            "led10min": False,
                            "led30min": False,
                            "led1hour": True
                        }
                    else:  # default
                        thingsboard_led_measure_period = {
                            "led10min": True,
                            "led30min": False,
                            "led1hour": False
                        }

                    # Telegram notification update
                    telegram_notifications_flag = raspberrypi3B.telegram_notifications
                    thingsboard_telegram_notifications = {"state_Telegram_switch": telegram_notifications_flag}
                    print(f"telegram_notifications_flag: {telegram_notifications_flag}")

                    thingsboard_message = {}
                    thingsboard_message.update(thingsboard_imminent_audio_request)
                    thingsboard_message.update(thingsboard_time_to_audio_request)
                    thingsboard_message.update(thingsboard_leds)
                    thingsboard_message.update(thingsboard_switches)
                    thingsboard_message.update(thingsboard_led_measure_period)
                    thingsboard_message.update(thingsboard_telegram_notifications)
                    payload = json.dumps(thingsboard_message)

                    # Publish data to the device telemetry topic
                    result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                    # Check if the publish was successful
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print("Data sent:", payload)
                    else:
                        print("Failed to send data. Return code:", result.rc)

                    iter_dashboard = 0
                else:
                    if raspberrypi3B.current_timer <= (raspberrypi3B.time_between_a_series_of_audio_requests-raspberrypi3B.caution_time_period_before_audio_request) and raspberrypi3B.imminent_audio_request_led_flag_thingsboard is True:
                        print("if iter_dashboard >= upload_dashboard_period and raspberrypi3B.imminent_audio_request_led_flag_thingsboard is True:")
                        # UPDATE IMMINENT AUDIO REQUEST LED
                        thingsboard_imminent_audio_request = {"imminent_audio_request_led": True}
                        # UPDATE TIME TO AUDIO REQUEST
                        time_to_audio_request = int(raspberrypi3B.time_between_a_series_of_audio_requests - raspberrypi3B.current_timer)
                        thingsboard_time_to_audio_request = {"time_to_audio_request": time_to_audio_request}
                        thingsboard_message = {}
                        thingsboard_message.update(thingsboard_imminent_audio_request)
                        thingsboard_message.update(thingsboard_time_to_audio_request)
                        payload = json.dumps(thingsboard_message)
                        print(f"payload: {payload}")
                        # Publish data to the device telemetry topic
                        result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                        # Check if the publish was successful
                        if result.rc == mqtt.MQTT_ERR_SUCCESS:
                            print("Data sent:", payload)
                        else:
                            print("Failed to send data. Return code:", result.rc)
                        raspberrypi3B.imminent_audio_request_led_flag_thingsboard = False
                    iter_dashboard += 1

                ####################################################################################

                # Check if it's time to request the audios, and if it's the case, start
                if raspberrypi3B.current_timer >= raspberrypi3B.time_between_a_series_of_audio_requests:
                    print("if raspberrypi3B.current_timer >= raspberrypi3B.time_between_a_series_of_audio_requests:")
                    print(f"if {raspberrypi3B.current_timer} >= {raspberrypi3B.time_between_a_series_of_audio_requests}:")
                    iter_dashboard = 0

                    print(f"raspberrypi3B.devices_to_make_an_audio_request_list: {raspberrypi3B.devices_to_make_an_audio_request_list}")

                    # UPDATE TIME TO AUDIO REQUEST
                    payload = json.dumps({"time_to_audio_request": 0})
                    print(f"payload: {payload}")
                    # Publish data to the device telemetry topic
                    result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                    # Check if the publish was successful
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print("Data sent:", payload)
                    else:
                        print("Failed to send data. Return code:", result.rc)


                    list_of_devices_id_to_put_to_sleep = raspberrypi3B.get_list_of_devices_id_to_put_to_sleep()
                    print(f"BEFORE DELETION: raspberrypi3B.devices_to_make_an_audio_request_list: {raspberrypi3B.devices_to_make_an_audio_request_list}")

                    for device_id in list_of_devices_id_to_put_to_sleep:
                        print("for device_id in list_of_devices_id_to_put_to_sleep:")
                        if device_id in raspberrypi3B.devices_to_make_an_audio_request_list:
                            print("if device_id in raspberrypi3B.devices_to_make_an_audio_request_list:")
                            raspberrypi3B.devices_to_make_an_audio_request_list.remove(device_id)
                            print(f"removed device_id: {device_id}")
                            # raspberrypi3B.devices_infos_dict.remove(device_id)
                            # raspberrypi3B.delete_device_from_the_net(device_id)
                    '''
                    for device_id in raspberrypi3B.devices_to_make_an_audio_request_list:
                        print("for device_id in raspberrypi3B.devices_to_make_an_audio_request_list:")
                        print(f"device_id: {device_id}")
                        if device_id in list_of_devices_id_to_put_to_sleep:
                            print("if device_id in list_of_devices_id_to_put_to_sleep:")
                            raspberrypi3B.devices_to_make_an_audio_request_list.remove(device_id)
                            print(f"removed device_id: {device_id}")
                            #raspberrypi3B.devices_infos_dict.remove(device_id)
                            #raspberrypi3B.delete_device_from_the_net(device_id)
                    '''

                    print(f"AFTER DELETION: raspberrypi3B.devices_to_make_an_audio_request_list: {raspberrypi3B.devices_to_make_an_audio_request_list}")
                    print(f"raspberrypi3B.devices_infos_dict: {raspberrypi3B.devices_infos_dict}")


                    # If there are some audios available, start requesting these audios to the related devices
                    if len(raspberrypi3B.devices_to_make_an_audio_request_list) > 0:
                        print("if len(raspberrypi3B.devices_to_make_an_audio_request_list) > 0:")

                        os.makedirs(f"received_audio_files", exist_ok=True)  # creating the folder (if not already present) where to save the audio files requested
                        for device_id in raspberrypi3B.devices_to_make_an_audio_request_list:
                            print_debug(f"for device_id in raspberrypi3B.devices_to_make_an_audio_request_list: -->device_id={device_id}")
                            raspberrypi3B.current_handled_device_id = device_id
                            audio_data_topic_key = generate_random_code(19) + '-'
                            if time.time() - raspberrypi3B.last_audio_request_publish_time >= raspberrypi3B.time_between_a_series_of_audio_requests:
                                time.sleep(2)  # just to be sure the esp32 task deletes itself before asking for the audio
                            request_audio(client, raspberrypi3B, device_id, audio_data_topic_key)
                            raspberrypi3B.go_on_with_audio_requests = False  # this variable will change whenever we receive and save the requested audio through the callback
                            sending_request_time = time.time()
                            while raspberrypi3B.go_on_with_audio_requests is False:  # I'll wait for doing other requests till the current requested audio has been received and saved properly
                                print_debug("while raspberrypi3B.go_on_with_audio_requests is False")
                                time.sleep(0.5)
                                current_time_request = time.time()
                                device_deleted_from_devices_infos_dict = False
                                if current_time_request - sending_request_time > 10:  # if I don't get the audio payload in less than 10 seconds, I retry with the request
                                    print_debug("   if current_time_request - sending_request_time > 10:")
                                    if device_deleted_from_devices_infos_dict is True:
                                        print("if device_deleted_from_devices_infos_dict is True:")
                                        break  # we don't continue to check the availability of the current device, since it has been previously deleted
                                    # client.unsubscribe("audio/payload" + audio_data_topic_key)
                                    # audio_data_topic_key = generate_random_code(20)
                                    # request_audio(client, raspberrypi3B, device_id, audio_data_topic_key)
                                    sending_request_time = time.time()
                                    waiting_time_response = sending_request_time
                                    while raspberrypi3B.dev_is_connected is False and device_deleted_from_devices_infos_dict is False:
                                        print_debug("         while raspberrypi3B.dev_is_connected is False and device_deleted_from_devices_infos_dict is False:")
                                        communication_id = generate_random_code(20)
                                        raspberrypi3B.communication_id = communication_id
                                        message_to_get_infos = {"device_id": device_id, "communication_id": communication_id}
                                        message_to_get_infos = json.dumps(message_to_get_infos)
                                        local_time = time.localtime(time.time())
                                        # Format struct_time to a string
                                        formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
                                        print_debug("+++++++++++++++++++++++++++++")
                                        print_debug(f"formatted_time: {formatted_time}")
                                        print_debug(f"topic: NodeMCU_get_infos")
                                        print_debug(f"Sending: {message_to_get_infos}")
                                        encrypted_message_to_get_infos = encrypt(message_to_get_infos)
                                        client.publish("NodeMCU_get_infos", encrypted_message_to_get_infos, qos=2) # client
                                        # time.sleep(5) # waiting a reasonable amount of time to get a response
                                        while True:
                                            print("while True:")
                                            local_time = time.localtime(time.time())
                                            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
                                            print_debug(f"formatted_time: {formatted_time}")
                                            time.sleep(1)
                                            if raspberrypi3B.dev_is_connected is True:
                                                print("if raspberrypi3B.dev_is_connected is True:")
                                                break
                                            current_time_waiting = time.time()
                                            if current_time_waiting - waiting_time_response > 5:
                                                print_debug("             if waiting_time_response - current_time_waiting > 5:")
                                                raspberrypi3B.delete_device_from_the_net(device_id)
                                                device_deleted_from_devices_infos_dict = True
                                                thingsboard_leds = {}
                                                list_of_devices = []
                                                for device_id, info in raspberrypi3B.devices_infos_dict.items():
                                                    location = raspberrypi3B.devices_infos_dict[device_id]['location']
                                                    hive_number = raspberrypi3B.get_hive_number_from_location(location)
                                                    list_of_devices.append(hive_number)

                                                max_hive_number = raspberrypi3B.num_max_allowed_devices
                                                if len(list_of_devices) > 0:
                                                    max_hive_number = max(list_of_devices)
                                                    if max_hive_number > raspberrypi3B.max_hive_number_ever_registered:
                                                        raspberrypi3B.max_hive_number_ever_registered = max_hive_number

                                                for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):
                                                    key_led = "ledStatus" + str(i)
                                                    # if i not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep() and i in list_of_devices:
                                                    if i in list_of_devices:
                                                        thingsboard_leds.update({key_led: 1})
                                                    else:
                                                        thingsboard_leds.update({key_led: 0})

                                                payload = json.dumps(thingsboard_leds)
                                                # Publish data to the device telemetry topic
                                                result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                                                # mosquitto_pub -d -q 1 -h localhost -p 1883 -t v1/devices/me/telemetry -u "tGVWWbIrFr9PDDDxmUwJ" -m "{temperature:25}"

                                                # Check if the publish was successful
                                                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                                    print("Data sent:", payload)
                                                else:
                                                    print("Failed to send data. Return code:", result.rc)


                                                break  # exit from this while loop


                                    if raspberrypi3B.dev_is_connected is True:  # if I exit from the inner while loop because raspberrypi3B.dev_is_connected becomes True, I reset the raspberrypi3B.dev_is_connected to False
                                        print_debug("         if raspberrypi3B.dev_is_connected is True:")
                                        raspberrypi3B.dev_is_connected = False  # reset for another audio request
                                        break  # ***************************
                                    if device_deleted_from_devices_infos_dict is True:  # if I deleted the current device_id since it's no more part of the network (we didn't get a valid response from it),
                                        # we exit from the more external while loop, and we continue requesting the audio for the successive device_id
                                        print_debug("         if device_deleted_from_devices_infos_dict is True:")
                                        break  # exit from the most external while loop

                            # Get the current time in microseconds
                            local_time = time.localtime(time.time())
                            # Format struct_time to a string
                            formatted_time = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
                            microseconds = int((time.time() % 1) * 1_000_000)
                            # Combine formatted time with microseconds
                            formatted_time_with_micros = f"{formatted_time}-{microseconds:06d}"
                            print_debug(f"{formatted_time_with_micros}")
                            client.unsubscribe("audio/payload" + audio_data_topic_key)  # no more audios sent to this topic cannot be accepted

                        # after the for loop above, all the requested audio has been saved into raspberrypi3B.receive_audios_folder_path




                        # Get a list of all filenames in the folder
                        if not os.path.exists(raspberrypi3B.receive_audios_folder_path):
                            os.makedirs(raspberrypi3B.receive_audios_folder_path)
                        filenames = os.listdir(raspberrypi3B.receive_audios_folder_path)
                        print(f"Filenames before preprocessing: {filenames}")
                        if raspberrypi3B.split_3sec_audio_flag is True:
                            # Splitting the received audios that reside inside the input folder in segment files of 1 sec each, and these will be saved inside the output folder,
                            # and every original audio from which we obtained a certain group of segment files will be deleted (leaving only the segment files)
                            input_folder = raspberrypi3B.receive_audios_folder_path
                            output_folder = raspberrypi3B.receive_audios_folder_path
                            split_audio_into_segments(input_folder, output_folder, filenames, segment_duration_sec=1)


                        # reset the list of devices that are ready to send the audio
                        raspberrypi3B.devices_to_make_an_audio_request_list = []

                        # This will hold the functions to be called
                        preprocess_and_obtain_npy_spectrograms_from_audio_folder_functions = [
                            preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread1,
                            preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread2,
                            preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread3,
                            preprocess_and_obtain_npy_spectrograms_from_audio_folder_thread4]

                        # move_files(source, destination)
                        num_audio_files, list_of_audio_in_the_order_as_they_appear_in_the_folder = count_wav_files_and_list_them(
                            raspberrypi3B.receive_audios_folder_path, raspberrypi3B.receive_audios_folder_path)
                        print_debug(
                            f"num_audio_files:{num_audio_files},list_of_audio_in_the_order_as_they_appear_in_the_folder:{list_of_audio_in_the_order_as_they_appear_in_the_folder}")

                        if num_audio_files > 0:
                            print_debug(f"num_audio_files: {num_audio_files}")
                            quotient = math.floor(
                                num_audio_files / 4)  # quotient is the number of files to preprocess per thread (there are 4 threads in the quad-core raspberrypy)
                            remainder = num_audio_files % 4  # remainder is the mod of num_audio_files mod 4
                            print_debug(f"quotient:{quotient}, remainder:{remainder}")
                            list_of_audios_to_preprocess_with_thread = []

                            '''
                            Example of balanced subdivision of the audio files per thread/core (quad-core raspberry-pi)

                            num_audios          TH1     TH2     TH3     TH4     quotient    remainder
                            __________________________________________________________________________
                            1                   1                               0           1
                            2                   1       2                       0           2
                            3                   1       2       3               0           3
                            4                   1       2       3       4       1           0
                            5                   1,2     3       4       5       1           1
                            6                   1,2     3,4     5       6       1           2
                            7                   1,2     3,4     5,6     7       1           3
                            8                   1,2     3,4     5,6     7,8     2           0
                            9                   1,2,3   4,5     6,7     8,9     2           1
                            10                  1,2,3   4,5,6   7,8     9,10    2           2
                            11                  1,2,3   4,5,6   7,8,9   10,11   2           3
                            ...........................................................................
                            ...........................................................................
                            ...........................................................................

                            '''

                            # PREPROCESSING SUBDIVISION FOR THE 4 THREADS/CORES
                            if remainder == 0:
                                list_of_audios_to_preprocess_with_thread.insert(0,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                0:quotient - 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(1,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                quotient:2 * quotient - 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(2,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                2 * quotient:3 * quotient - 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(3,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                3 * quotient:4 * quotient - 1 + 1])
                            elif remainder == 1:
                                print_debug(
                                    f"list_of_audio_in_the_order_as_they_appear_in_the_folder[0:quotient+1]: {list_of_audio_in_the_order_as_they_appear_in_the_folder[0:quotient + 1]}")
                                list_of_audios_to_preprocess_with_thread.insert(0,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                0:quotient + 1])
                                list_of_audios_to_preprocess_with_thread.insert(1,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                quotient + 1:2 * quotient + 1])
                                list_of_audios_to_preprocess_with_thread.insert(2,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                2 * quotient + 1:3 * quotient + 1])
                                list_of_audios_to_preprocess_with_thread.insert(3,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                3 * quotient + 1:4 * quotient + 1])
                            elif remainder == 2:
                                list_of_audios_to_preprocess_with_thread.insert(0,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                0:quotient + 1])
                                list_of_audios_to_preprocess_with_thread.insert(1,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                quotient + 1:2 * quotient + 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(2,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                2 * quotient + 2:3 * quotient + 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(3,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                3 * quotient + 2:4 * quotient + 1 + 1])
                            elif remainder == 3:
                                list_of_audios_to_preprocess_with_thread.insert(0,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                0:quotient + 1])
                                list_of_audios_to_preprocess_with_thread.insert(1,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                quotient + 1:2 * quotient + 1 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(2,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                2 * quotient + 2:3 * quotient + 2 + 1])
                                list_of_audios_to_preprocess_with_thread.insert(3,
                                                                                list_of_audio_in_the_order_as_they_appear_in_the_folder[
                                                                                3 * quotient + 3:4 * quotient + 2 + 1])

                            threads = []

                            print_debug(
                                f"list_of_audios_to_preprocess_with_thread: {list_of_audios_to_preprocess_with_thread}")
                            for i in range(0, 4):
                                # thread = threading.Thread(target=preprocess_and_obtain_npy_spectrograms_from_audio_folder_functions[i], args=f"received_audio_files/thread{i}/")
                                if len(list_of_audios_to_preprocess_with_thread[i]) != 0:
                                    thread = threading.Thread(
                                        target=preprocess_and_obtain_npy_spectrograms_from_audio_folder_functions[i],
                                        args=(raspberrypi3B.receive_audios_folder_path,
                                              list_of_audios_to_preprocess_with_thread[i]))
                                    threads.append(thread)
                                    thread.start()

                            # Wait for all threads to complete
                            for thread in threads:
                                thread.join()







                            '''
                            # IT'S NOT POSSIBLE TO GO MULTITHREADING FOR RUNNING THE INFERENCE, SINCE THE ARMNN CODE ITSELF ALREADY USES THE 4 CORES. THUS THE CODE BELOW DOESN'T WORK.
                            run_cpp_executable_functions = [run_cpp_executable1, run_cpp_executable2, run_cpp_executable3, run_cpp_executable4]
                            threads = []

                            for i in range(0,4):
                                thread = threading.Thread(target=run_cpp_executable_functions[i])
                                threads.append(thread)
                                thread.start()

                            # Wait for all threads to complete
                            for thread in threads:
                                thread.join()
                            '''
                            # RUNNING THE INFERENCE COMMAND (SUBPROCESS)
                            run_cpp_executable_for_inference_classification()

                            # Classification queen/noqueen
                            classify_audios_by_labels_and_update_state(filenames, raspberrypi3B.receive_audios_folder_path, raspberrypi3B)






                            ##############################################################################################################
                            # Writing in the .csv file
                            import csv
                            # Define the file name
                            file_name = 'History.csv'

                            # Define the header in the desired order
                            header = ['location', 'state', 'fixed_id', 'device_id', 'device_model', 'battery_level',
                                      'ip_address', 'enable_audio_record', 'last_record_date']

                            # Prepare the data to be written to the CSV file
                            data_to_write = []
                            thingsboard_data = {}
                            thingsboard_alarms = []
                            alarm_notification = False
                            telegram_message = "No queen detected at the hives: "
                            for device_id, info in raspberrypi3B.devices_infos_dict.items():
                                # Extract information from the dictionary, maintaining the order
                                row = [
                                    info['location'],
                                    info['state'],
                                    info['fixed_id'],
                                    device_id,
                                    info['device_model'],
                                    info['battery_level'],
                                    info['ip_address'],
                                    info['enable_audio_record'],
                                    info['last_record_date']
                                ]
                                data_to_write.append(row)

                                if info['state'] == "noqueen":
                                    alarm_notification = True
                                    telegram_message += info['location'] + ','
                                    thingsboard_alarms.append({"Device_model_alarm":info['device_model'], "Hive_alarm":info['location'], "State_alarm":info['state']})


                                print(f"thingsboard_alarms: {thingsboard_alarms}")
                                thingsboard_hive, thingsboard_state = raspberrypi3B.get_thingsboard_values(device_id)
                                thingsboard_data.update({thingsboard_hive: thingsboard_state})




                            print(f"thingsboard_data: {thingsboard_data}")

                            # Open the CSV file in append mode
                            with open(file_name, mode='a', newline='') as file:
                                writer = csv.writer(file)

                                # Check if the file is empty to write the header only when necessary
                                file.seek(0, 2)  # Move the cursor to the end of the file
                                if file.tell() == 0:  # If the size of the file is 0, write the header
                                    writer.writerow(header)

                                # Write the prepared data
                                writer.writerows(data_to_write)

                            # The file is automatically closed after exiting the with statement
                            print(f"Data has been appended to {file_name} successfully.")
                            ##############################################################################################################

                            # TELEGRAM NOTIFICATION
                            if alarm_notification is True and raspberrypi3B.telegram_notifications is True:
                                result = send_telegram_message(CHAT_ID, telegram_message[:-1], BOT_TOKEN)
                                print(result)




                            # UPDATE THINGSBOARD
                            thingsboard_message = {}

                            # Thingsboard update state charts
                            thingsboard_message.update(thingsboard_data)

                            # UPDATE IMMINENT AUDIO REQUEST LED
                            thingsboard_message.update({"imminent_audio_request_led": False})

                            # UPDATE TIME TO AUDIO REQUEST
                            thingsboard_message.update({"time_to_audio_request": raspberrypi3B.time_between_a_series_of_audio_requests})

                            payload = json.dumps(thingsboard_message)
                            print(f"payload: {payload}")
                            # Publish data to the device telemetry topic
                            result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                            # Check if the publish was successful
                            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                print("Data sent:", payload)
                            else:
                                print("Failed to send data. Return code:", result.rc)


                            # THINGSBOARD ALARM STATUS
                            if alarm_notification is True and len(thingsboard_alarms) != 0:
                                print("if alarm_notification is True and len(thingsboard_alarms) != 0:")
                                for alarm_dict in thingsboard_alarms:
                                    print("for alarm_dict in thingsboard_alarms:")
                                    print(f"alarm_dict: {alarm_dict}")
                                    payload = json.dumps(alarm_dict)
                                    print(f"payload: {payload}")
                                    # Publish data to the device telemetry topic
                                    result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                                    # Check if the publish was successful
                                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                        print("Data sent:", payload)
                                    else:
                                        print("Failed to send data. Return code:", result.rc)
                                    time.sleep(1)


                            #raspberrypi3B.subscription_in_progress = False # If for example a subscription attempt didn't end in a received ack by the esp32, subscription_in_progress could remain as True indefinitely,
                                                                           # thus there is the risk other devices can't subscribe (since a device can subscribe only if subscription_in_progress == False)













                            time.sleep(2)
                            # Moving all the files contained in the receive_audios_folder_path to the collect_audios_folder_path, so that for the next series of audio requests, the receive_audios_folder_path is properly empty
                            move_files(raspberrypi3B.receive_audios_folder_path,
                                       raspberrypi3B.collect_audios_folder_path)  # move the files from the previous audio_folder to the collector_folder, in order to empty the audio_folder for the next preprocess operations

                        # Update the last_audio_request_publish_time (to be used at the beginning of the main, in order to check if it's time to start requesting the audios)
                        raspberrypi3B.last_audio_request_publish_time = time.time()



                    if len(raspberrypi3B.last_communication_epoch_dict) != 0:
                        print("if len(raspberrypi3B.last_communication_epoch_dict) != 0:")
                        current_epoch = time.time()
                        device_id_to_delete_list = []
                        for device_id in raspberrypi3B.last_communication_epoch_dict:
                            print("for device_id in raspberrypi3B.last_communication_epoch_dict:")
                            print(f"device_id: {device_id}")
                            last_communication_epoch_with_the_device = raspberrypi3B.last_communication_epoch_dict[
                                device_id]
                            print(
                                f"last_communication_epoch_with_the_device: {last_communication_epoch_with_the_device}")
                            print(
                                f"{current_epoch - last_communication_epoch_with_the_device} >= {raspberrypi3B.max_time_without_communication_with_a_device} ????")
                            if current_epoch - last_communication_epoch_with_the_device >= raspberrypi3B.max_time_without_communication_with_a_device:
                                print(
                                    "if current_epoch - last_communication_epoch_with_the_device >= raspberrypi3B.max_time_without_communication_with_a_device:")
                                device_id_to_delete_list.append(device_id)
                                print(f"device_id_to_delete_list: {device_id_to_delete_list}")
                        for device_id in device_id_to_delete_list:
                            print("for device_id in device_id_to_delete_list:")
                            raspberrypi3B.delete_device_from_the_net(device_id)
                            thingsboard_leds = {}
                            list_of_devices = []
                            for device_id, info in raspberrypi3B.devices_infos_dict.items():
                                location = raspberrypi3B.devices_infos_dict[device_id]['location']
                                hive_number = raspberrypi3B.get_hive_number_from_location(location)
                                list_of_devices.append(hive_number)

                            max_hive_number = raspberrypi3B.num_max_allowed_devices
                            if len(list_of_devices) > 0:
                                max_hive_number = max(list_of_devices)
                                if max_hive_number > raspberrypi3B.max_hive_number_ever_registered:
                                    raspberrypi3B.max_hive_number_ever_registered = max_hive_number

                            for i in range(1, raspberrypi3B.max_hive_number_ever_registered + 1):
                                key_led = "ledStatus" + str(i)
                                # if i not in raspberrypi3B.get_list_of_hives_numbers_to_put_to_sleep() and i in list_of_devices:
                                if i in list_of_devices:
                                    thingsboard_leds.update({key_led: 1})
                                else:
                                    thingsboard_leds.update({key_led: 0})

                            payload = json.dumps(thingsboard_leds)
                            # Publish data to the device telemetry topic
                            result = client_thingsboard.publish(f'v1/devices/me/telemetry', payload, qos=1)
                            # mosquitto_pub -d -q 1 -h localhost -p 1883 -t v1/devices/me/telemetry -u "tGVWWbIrFr9PDDDxmUwJ" -m "{temperature:25}"

                            # Check if the publish was successful
                            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                print("Data sent:", payload)
                            else:
                                print("Failed to send data. Return code:", result.rc)

                # else:  # waiting for the time to pass in order to do another series of audio requests
                #    pass

                ## Process MQTT events (including handling callbacks)
                # client.loop()  # This will process network events, including any incoming messages and callbacks

                # Optional: Sleep for a short time to reduce CPU usage
                time.sleep(1)

        except KeyboardInterrupt:
            print_debug("Exiting...")
            client.loop_stop()  # Stop the loop gracefully
            client.disconnect()  # Disconnect from the broker

    except Exception as e:
        print_debug(f"An error occurred in main(): {e}")
        client.loop_stop()  # Ensure the loop is stopped
        client.disconnect()  # Ensure the client is disconnected

    ########################################################################################

    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time
    print_debug(f"Elapsed time: {elapsed_time:.2f} seconds")


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


