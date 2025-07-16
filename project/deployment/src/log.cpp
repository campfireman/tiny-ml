#include <Arduino.h>

#include "log.hpp"

void logDebug()
{
}
void logDebugln()
{
}

#include "esp_heap_caps.h"

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

    // Extended SRAM info
    multi_heap_info_t heapInfo;
    heap_caps_get_info(&heapInfo, MALLOC_CAP_INTERNAL); // internal RAM (SRAM)

    Serial.println("--- Internal SRAM Info ---");
    Serial.print("Total Allocated: ");
    Serial.println(heapInfo.total_allocated_bytes);
    Serial.print("Total Free: ");
    Serial.println(heapInfo.total_free_bytes);
    Serial.print("Largest Free Block: ");
    Serial.println(heapInfo.largest_free_block);

    // Optional: stack info (only useful if you are checking task-level usage)
    Serial.print("Free Stack (main task): ");
    Serial.println(uxTaskGetStackHighWaterMark(NULL));
}
