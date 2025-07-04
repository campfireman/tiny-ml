#include <WiFi.h>
#include <WiFiMulti.h>

#include "secrets.h"
#include "wifi.hpp"

WiFiMulti WiFiMulti;

void wifiInit()
{
    // We start by connecting to a WiFi network
    WiFiMulti.addAP(WIFI_SSID, WIFI_PASS);

    Serial.println();
    Serial.println();
    Serial.print("Waiting for WiFi... ");

    while (WiFiMulti.run() != WL_CONNECTED)
    {
        Serial.print(".");
        delay(500);
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}
