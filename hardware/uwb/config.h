#pragma once

// WiFi
inline const char *ssid = "BEL 7462";
inline const char *password = "9*9V7p68";

// MQTT
inline const char *mqtt_broker = "172.20.10.11";
inline const char *clientID = "esp32-player-client";
inline const std::string playerEspPublishTopic = "/playerPosition";
inline const std::string playerEspSubscribeTopics[] = {"/system/signal", "/positionCalibration"};


// TLS Configuration
inline const char* caCert = "";
inline const char* clientCert = "";
inline const char* clientKey = "";