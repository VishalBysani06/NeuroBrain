#include <Servo.h>

Servo myservo;
int emgPin = A0;   // Analog pin for EMG sensor
int emgValue = 0;  // EMG sensor value
int servoPin = 9;  // Servo pin

void setup() {
  myservo.attach(servoPin);  // Attach servo to pin 9
  Serial.begin(9600);        // Initialize serial communication
}

void loop() {
  emgValue = analogRead(emgPin);                  // Read EMG sensor value
  int servoPos = map(emgValue, 0, 1023, 0, 180);  // Map EMG sensor value to servo position
  myservo.write(servoPos);                        // Set servo position
  Serial.println(emgValue);                       // Print EMG sensor value
  delay(100);                                     // Wait for 100ms
}
