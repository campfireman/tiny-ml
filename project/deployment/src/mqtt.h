void mqttInit();
void mqttConnect();
void mqttSend(String command);
void mqttRefresh();
void publishSensorData(String measurement, String displayName, String unit, String deviceClass, String data);
