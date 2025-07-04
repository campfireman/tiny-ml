#include <Arduino.h>

#include "log.hpp"

void logDebug()
{
}
void logDebugln()
{
}

void printHeapInfo()
{
    Serial.print("Heap: ");
    Serial.println(ESP.getHeapSize());
    Serial.print("Free Heap: ");
    Serial.println(ESP.getFreeHeap());
    Serial.print("PSRAM: ");
    Serial.println(ESP.getPsramSize());
    Serial.print("Free PSRAM: ");
    Serial.println(ESP.getFreePsram());
}
