TEST INFOS TO CHECK IF THE DEV BOARD WORKS CORRECTLY








//MACRO DEFINITIONS


// RGB LED
#define PIN_RED    33 //Please do not modify it
#define PIN_GREEN  26 //Please do not modify it
#define PIN_BLUE   27 //Please do not modify it

// ENABLES
#define SD_CARD_3V3_ENABLE 4 //Please do not modify it
#define H2_H3_3V3_ENABLE 18 //Please do not modify it. In the header H2 is also present the 3V3 of the microphone
#define I2C_J1_3V3_ENABLE 5 //Please do not modify it
#define BATTERY_MONITOR_ENABLE 25 //Please do not modify it

// BATTERY MONITOR ADC PORT
#define ADC_PORT_BATTERY_MONITOR 34 //Please do not modify it

// TEST MICRO SD-CARD AND MICROPHONE ENABLE
// If MICROPHONE ENABLED, MICRO SD-CARD ENABLED; --> FILE SUCCESSFULLY WRITTEN IN THE MICRO SD-CARD
// If MICROPHONE ENABLED, MICRO SD-CARD DISABLED; --> THE MICRO SD-CARD WILL NOT BE FOUND, AND THE ESP32 WILL REBOOT
// If MICROPHONE DISABLED, MICRO SD-CARD ENABLED; --> THE FILE WILL NOT BE SUCCESSFULLY WRITTEN
// If MICROPHONE DISABLED, MICRO SD-CARD DISABLED; --> THE MICRO SD-CARD WILL NOT BE FOUND, AND THE ESP32 WILL REBOOT
#define LOGIC_SD_CARD_3V3_ENABLE_VALUE HIGH // WHEN HIGH, THE MICRO SD-CARD WILL WORK, VICEVERSA WHEN LOW
#define LOGIC_H2_H3_3V3_ENABLE_VALUE HIGH // WHEN HIGH, THE MICROPHONE WILL WORK, VICEVERSA WHEN LOW


#define TIME_BETWEEN_RECORDINGS 10 //seconds between consecutive recordings 

// MICRO SD-CARD DEFINITIONS
#define SD_MMC_CMD 15 //Please do not modify it
#define SD_MMC_CLK 14 //Please do not modify it
#define SD_MMC_D0  2  //Please do not modify it








									TESTS TO DO:
									





TEST USB-C
	UPLOAD THE CODE
	THE BOARD SHOULD RESULT SUPPLIED BY THE USB-C (CHECK THE VOLTAGE AT TP2 AND TP5)
_____________________________________________________________________________________
TEST BATTERY
	TURN THE SWITCH SW3 ON-OFF TO CONNECT/DISCONNECT THE BATTERY
	THE BOARD SHOULD RESULT SUPPLIED BY THE BATTERY (CHECK THE VOLTAGE AT TP2)
_____________________________________________________________________________________
TEST USB-C + BATTERY
	THE BOARD SHOULD RESULT SUPPLIED BY THE USB-C (CHECK THE VOLTAGE AT TP2 AND TP5)
_____________________________________________________________________________________
TEST BUCK-BOOST OUTPUT VOLTAGE
	CHECK TP3 --> 3.3V VALUE
_____________________________________________________________________________________
TEST MICRO-SD_CARD (SD_CARD_3V3_ENABLE GPIO4)
	SD_CARD_3V3_ENABLE HIGH ---> ON
	SD_CARD_3V3_ENABLE LOW  ---> OFF
_____________________________________________________________________________________
TEST MICROPHONE, H1 AND H2_H3_3V3 LINES (H2_H3_3V3_ENABLE GPIO18)
	H2_H3_3V3_ENABLE HIGH ---> ON
	H2_H3_3V3_ENABLE LOW  ---> OFF
	IF THE MICROPHONE WORKS, THEN GPIO2/14/15/12 WORK TOO, AND IF THE MICRO SD-CARD IS ENABLED, THE AUDIO FILE RECORDED WILL BE SUCCESSFULLY WRITTEN ON THE SD-CARD
_____________________________________________________________________________________
TEST BATTERY MONITOR (BATTERY_MONITOR_ENABLE GPIO25, ADC_PORT_BATTERY_MONITOR GPIO34)
	CHECK VBATT/2 ON TP4 DOING:
		BATTERY_MONITOR_ENABLE HIGH ---> ON (THE VOLTAGE WILL BE VBATT/2 ON TP4) ---> THE RGB LED WILL BE WATER-GREEN COLOR DURING THE ADC READING
		BATTERY_MONITOR_ENABLE LOW  ---> OFF (THE VOLTAGE WILL BE 0V ON TP4) ---> THE RGB LED WILL STAY OFF
_____________________________________________________________________________________
TEST RTC
	INSERT THE CR1220 3V BATTERY AND SAVE THE AUDIO FILES WITH THE CURRENT DATE IN THEIR NAMES
_____________________________________________________________________________________
TEST I2C_J1_3V3 LINE (I2C_J1_3V3_ENABLE GPIO5)
	INSERT THE CR1220 3V BATTERY INTO THE RTC
	I2C_J1_3V3_ENABLE HIGH ---> ON
	I2C_J1_3V3_ENABLE LOW ---> OFF
_____________________________________________________________________________________
TEST I2C_J2
	INSERT THE CR1220 3V BATTERY INTO THE RTC
	CONNECT SOME I2C CABLES
_____________________________________________________________________________________
TEST H5
	CHECK GND AND 5V (FOR THESE JUST CHECK THE VOLTAGES AT TP1 AND TP5 RESPECTIVELY)
_____________________________________________________________________________________
TEST H3
	3V3 WILL BE AVAILABLE ONLY IF H2_H3_3V3_ENABLE IS ON
	CHECK THE TWO GND LINES
	CHECK GPIO13
	CHECK GPIO32
	CHECK GPIO35 (ONLY INPUT)
_____________________________________________________________________________________
TEST H4
	CHECK 3V3
	CHECK THE TWO GND LINES
	CHECK IO0 (DO NOT PRESS THE SW1 BUTTON WHILE CHECKING)
	CHECK GPIO19
	CHECK GPIO23
_____________________________________________________________________________________
TEST H6 (DIFFERENTIAL INPUT)
	CHECK VP
	CHECK VN
_____________________________________________________________________________________
TEST RGB LED
	TURN-ON THE LED DOING:
		analogWrite(GPIO33, value),  FOR RED
		analogWrite(GPIO26, value),  FOR GREEN
		analogWrite(GPIO27, value),  FOR BLUE
		WHERE value CAN BE AN INTEGER NUMBER BETWEEN 0 and 255,
		BUT NOT 0 FOR ALL THE THREE GPIOs TOGETHER
	TURN-OFF THE LED DOING:
		analogWrite(GPIO33, 0),  FOR RED
		analogWrite(GPIO26, 0),  FOR GREEN
		analogWrite(GPIO27, 0),  FOR BLUE
_____________________________________________________________________________________








								On Arduino-IDE go to






File-->Preferences-->Additional boards manager URLs and insert:     https://dl.espressif.com/dl/package_esp32_index.json

as depicted in the image Screen "Arduino-IDE.png".



For the libraries used see the image "Libraries_installed.png".

For the board configurations, see the image "Board_Arduino-IDE_configuration.png".



AFTER UPLOADING THE TEST CODE, 'TEST RGB LED', TOGETHER WITH 'TEST H3' AND 'TEST H4' WILL START.
AFTER THAT, 10 SECONDS WILL ELAPSE TILL THE BATTERY VOLTAGE MONITORING (THE RGB LED WILL TURN WATER-GREEN DURING THE BATTERY MONITORING) AND ALSO A CONSECUTIVE AUDIO RECORDING WILL START.
THEN, THE NEXT BATTERY MONITORINGS + AUDIO RECORDINGS WILL BE DONE EVERY 10 SECONDS.
	
	
	


	


