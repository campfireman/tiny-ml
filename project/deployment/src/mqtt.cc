#include <Arduino.h>
#include <Arduino_JSON.h>
#include <HTTPClient.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <WiFiMulti.h>

#include "mqtt.h"
#include "secrets.h"

#define DEBUG true
#define MQTT_ERR -1

String mqttTopic = "";
String deviceID;
bool configPublished = false;

bool HOASS_AUTO_DISCOVERY = true;
unsigned long lastMqttConnectTry = 0;
const unsigned long mqttRetryInterval = 600000; // millis in 10 minutes
const int MQTT_DELAY = 6000;

WiFiClient client;
PubSubClient mqttclient(MQTT_HOST, MQTT_PORT, client);

String getNormalizedMac()
{
    String mac = WiFi.macAddress();
    mac.replace(":", "");
    mac.toLowerCase();
    return mac;
}

void mqttInit()
{
    /** WIFI connect */
    while (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("WiFi Client not available for MQTT, waiting...");
        delay(500);
    }
    deviceID = getNormalizedMac();
    mqttTopic = "voice/" + deviceID + "/";
    Serial.println("MQTT Topic: " + mqttTopic);

    if (!mqttclient.connect(deviceID.c_str(), MQTT_USERNAME, MQTT_PASSWORD, ("voice/" + deviceID + "/status").c_str(), 1, true, "offline"))
    {
        Serial.println("Failed to connect to MQTT server");
        delay(MQTT_DELAY);
        return;
    }
    mqttclient.publish(("voice/" + deviceID + "/status").c_str(), "online", true);
    Serial.println("Connected to MQTT server.");
    delay(1000);
    boolean bufferSizeIncreased = mqttclient.setBufferSize(512);
    if (!bufferSizeIncreased)
    {
        Serial.println("Failed to increase buffer size");
        if (HOASS_AUTO_DISCOVERY)
        {
            HOASS_AUTO_DISCOVERY = false;
            delay(5000);
            Serial.println("Disabling Home Assistant auto discovery");
        }
    }
}

void mqttRefresh()
{
    if (!mqttclient.connected())
    {
        mqttInit();
    }
    mqttclient.loop();
}

void mqttSend(String command)
{
    mqttRefresh();
    Serial.println("Attempting to send to MQTT...");

    if (WiFi.status() == WL_CONNECTED && mqttclient.connected())
    {
        publishSensorData("command", "Voice Command", "", "enum", command);
    }
    else
    {
        Serial.println("Not ready to send to MQTT: WIFI status " + String(WiFi.status()) + " and MQTT-client: " + mqttclient.connected());
    }
}

void publishSensorData(String measurement, String displayName, String unit, String deviceClass, String data)
{
    String publishTopic = String(mqttTopic) + measurement;

    mqttclient.publish(publishTopic.c_str(), data.c_str()); // Use data instead of int
    Serial.println("Published to topic: " + publishTopic + " data: " + data);

    if (HOASS_AUTO_DISCOVERY && !configPublished)
    {
        JSONVar doc;

        doc["name"] = displayName;
        doc["unit_of_measurement"] = unit;
        doc["state_topic"] = publishTopic;
        doc["device_class"] = deviceClass;
        doc["availability_topic"] = String("voice/") + deviceID + "/status";
        doc["unique_id"] = deviceID + "_" + measurement;
        JSONVar device;
        JSONVar identifiers;
        identifiers[0] = deviceID.c_str();
        device["name"] = "voice-command-detector" + deviceID;
        device["model"] = deviceID;
        device["manufacturer"] = "ture";

        device["identifiers"] = identifiers;
        doc["device"] = device;

        String payload = JSON.stringify(doc);
        String topic = "homeassistant/sensor/" + deviceID + "/" + measurement + "/config";
        mqttclient.publish(topic.c_str(), payload.c_str(), true);
        Serial.println("Published to topic: " + topic + " data: " + payload);
        configPublished = true;
    }
}
