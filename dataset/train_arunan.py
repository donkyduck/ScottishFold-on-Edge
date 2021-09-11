import pyshark
import sqlite3
import sys
from os import path
import base64
import time
import glob
sys.path.append(path.dirname(path.dirname(path.abspath(__file__) ) ) )
import os
from database.db_manager import closeConnection
TRAIN_PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_paper.db")

def createDatabase():
    try:
        sqliteConnection = sqlite3.connect(TRAIN_PAPER)
        cursorObject = sqliteConnection.cursor()
        createTable = "CREATE TABLE IF NOT EXISTS RESULT(ipSrc string(40), ipDst string(40), macSrc string(20), macDst string(20),"
        createTable += " portSrc string(10), portDst string(10), pktLength INTEGER, deviceName string(20), protocol string(10), detail TEXT UNIQUE)"
        cursorObject.execute(createTable)
    except:
        print "CAN'T CREATE TABLE :("
    finally:
        closeConnection(sqliteConnection)

def main():
    #protocolArray = ["tcp", "udp", "mqtt", "dns", "http", "ntp", "mdns" ,"ssl"]
    protocolArray = ["ntp","ssl"]
    pcapPathArray = fileList()
    for i in range (0, len(protocolArray)):
        protocol = str(protocolArray[i])
        print protocol
        for x in range(0, len(pcapPathArray)):
            cap = pyshark.FileCapture(pcapPathArray[x], display_filter=protocol)
            cap.load_packets()
            for order in range(0,len(cap)):
                currentCap = cap[order]
                tempProtocol = protocol.upper()
                try:
                    if str(currentCap.highest_layer) == str(tempProtocol):
                        protocolChk = str(currentCap.frame_info.protocols)
                        ipSrc,ipDst = getIP(currentCap)
                        macSrc = getMac(currentCap,protocolChk,"src")
                        macDst = getMac(currentCap,protocolChk,"dst")
                        portSrc = getPort(currentCap,protocolChk,"src")
                        portDst = getPort(currentCap,protocolChk,"dst")
                        deviceName = getDeviceName(macSrc,macDst)
                        pktLength = str(currentCap.length)
                        if protocol == "tcp":
                            detail = str(currentCap.tcp)
                            protocolID = "1"
                        elif protocol == "udp":
                            detail = str(currentCap.udp)
                            protocolID = "2"
                        elif protocol == "mqtt":
                            detail = str(currentCap.mqtt)
                            protocolID = "3"
                        elif protocol == "dns":
                            detail = str(currentCap.dns)
                            protocolID = "4"
                        elif protocol == "http":
                            detail = str(currentCap.http)
                            protocolID = "5"
                        elif protocol == "ntp":
                            detail = str(currentCap.ntp)
                            protocolID = "6"
                        elif protocol == "mdns":
                            detail = str(currentCap.mdns)
                            protocolID = "7"
                        elif protocol == "ssl":
                            detail = str(currentCap.ssl)
                            protocolID = "8"

                        # Check for deviceName before write to database.
                        if deviceName == "0":
                            print "SKIP!"
                        else:
                            pushToDB(ipSrc,ipDst,macSrc,macDst,portSrc,portDst,pktLength,deviceName,protocolID,detail)
                    else:
                        pass
                except:
                    print ":("
            cap.close()
    #except:
    #    print "error"
    #    pass

def pushToDB(ipSrc,ipDst,macSrc,macDst,portSrc,portDst,pktLength,deviceName,protocolID,detail):
    try:
        sqliteConnection = sqlite3.connect(TRAIN_PAPER)
        insertToTable = "INSERT INTO RESULT(ipSrc, ipDst, macSrc, macDst, portSrc, portDst, pktLength, deviceName, protocol, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        dataTuple = (ipSrc, ipDst, macSrc, macDst, portSrc, portDst, pktLength, deviceName, protocolID, detail)
        cursor = sqliteConnection.cursor()
        cursor.execute(insertToTable, dataTuple)
        sqliteConnection.commit()
        print(ipSrc + ", " + ipDst + ", " + macSrc + ", " + macDst + ", " + portSrc + ", " + portDst + ", " + pktLength + ", " + deviceName + ", "  + protocolID)
        cursor.close()
    except:
        print "DUPLICATE DATA :("
        pass
    finally:
        closeConnection(sqliteConnection)

def getMac(currentCap,protocolChk,srcOrDst):
    subsEthernet = "eth"
    subsWLAN = "wlan"
    if subsEthernet in protocolChk and srcOrDst == "src":
        srcMac = str(currentCap.eth.src)
        return srcMac
    elif subsEthernet in protocolChk and srcOrDst == "dst":
        dstMac = str(currentCap.eth.dst)
        return dstMac
    elif subsWLAN in protocolChk and srcOrDst == "src":
        srcMac = str(currentCap.wlan.sa)
        return srcMac
    elif subsWLAN in protocolChk and srcOrDst == "dst":
        dstMac = str(currentCap.wlan.da)
        return dstMac
    else:
        pass

def getPort(currentCap,protocolChk,srcOrDst):
    subsTCP = "tcp"
    subsUDP = "udp"
    if subsTCP in protocolChk and srcOrDst == "src":
        srcPort = str(currentCap.tcp.srcport)
        return srcPort
    elif subsTCP in protocolChk and srcOrDst == "dst":
        dstPort = str(currentCap.tcp.dstport)
        return dstPort
    elif subsUDP in protocolChk and srcOrDst == "src":
        srcPort = str(currentCap.udp.srcport)
        return srcPort
    elif subsUDP in protocolChk and srcOrDst == "dst":
        dstPort = str(currentCap.udp.dstport)
        return dstPort
    else:
        pass

def getDeviceName(macSrc,macDst):
    if "cc:50:e3:da:00:7f" in macSrc or "cc:50:e3:da:00:7f" in macDst:
        deviceName = "1" # Anitech
    elif "60:01:94:ac:93:31" in macSrc or "60:01:94:ac:93:31" in macDst:
        deviceName = "2" # Sonoff B1
    elif "54:e5:bd:8c:5c:5e" in macSrc or "54:e5:bd:8c:5c:5e" in macDst:
        deviceName = "3" # GuRobot
    elif "ac:84:c6:21:07:3e" in macSrc or "ac:84:c6:21:07:3e" in macDst:
        deviceName = "4" # TP-Link HS110
    elif "00:17:88:b2:6b:0c" in macSrc or "00:17:88:b2:6b:0c" in macDst:
        deviceName = "5" # Philips Hue Bridge
    elif "ac:84:c6:bf:fc:a5" in macSrc or "ac:84:c6:bf:fc:a5" in macDst:
        deviceName = "6" # TP-Link KC120
    elif "50:c7:bf:8d:87:b6" in macSrc or "50:c7:bf:8d:87:b6" in macDst:
        deviceName = "7" # TP-Link LB100
    elif "18:b4:30:8f:88:a8" in macSrc or "18:b4:30:8f:88:a8" in macDst:
        deviceName = "8" # Google Nest Cam IQ
    elif "60:01:94:74:22:a6" in macSrc or "60:01:94:74:22:a6" in macDst:
        deviceName = "9" # Sonoff S20
    elif "cc:50:e3:00:68:c8" in macSrc or "cc:50:e3:00:68:c8" in macDst:
        deviceName = "10" # Sonoff S26
    else:
        deviceName = "0"
    return deviceName

def getIP(currentCap):
    srcIP = str(currentCap.ip.src)
    dstIP = str(currentCap.ip.dst)
    return srcIP,dstIP

def fileList():
    fileInDirectory = glob.glob("/home/meowmeow/Desktop/Database/PAIR/*.cap")
    return fileInDirectory
# Run
#createDatabase()

main()
