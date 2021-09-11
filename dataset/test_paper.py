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
TRAIN_PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_paper.db")
MAC_OUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mac_oui.db")

def createDatabase():
    try:
        sqliteConnection = sqlite3.connect(TRAIN_PAPER)
        cursorObject = sqliteConnection.cursor()
        createTable = "CREATE TABLE IF NOT EXISTS RESULT(ipAddr string(40), macAddr string(20),"
        createTable += " portSrc string(10), portDst string(10), pktLength INTEGER, deviceName string(20), protocol string(10), detail TEXT)"
        cursorObject.execute(createTable)
    except:
        print "CAN'T CREATE TABLE :("
    finally:
        closeConnection(sqliteConnection)

def main():
    #protocolArray = ["tcp", "udp", "mqtt", "dns", "http", "ntp", "mdns" ,"ssl"]
    protocolArray = ["tcp","ssl"]
    pcapPathArray = fileList()
    for i in range (0, len(protocolArray)):
        protocol = str(protocolArray[i])
        print protocol
        for x in range(0, len(pcapPathArray)):
            cap = pyshark.FileCapture(pcapPathArray[x], display_filter=protocol)
            #cap = pyshark.FileCapture('/home/meowmeow/Desktop/Database/OTHER/7_8_2020.pcap')
            cap.load_packets()
            for order in range(0,len(cap)):
                currentCap = cap[order]
                tempProtocol = protocol.upper()
                try:
                    if str(currentCap.highest_layer) == str(tempProtocol):
                        protocolChk = str(currentCap.frame_info.protocols)
                        ipAddr,srcOrDst = getIP(currentCap)
                        macAddr = getMac(currentCap,protocolChk,srcOrDst)
                        portSrc = getPort(currentCap,protocolChk,"src")
                        portDst = getPort(currentCap,protocolChk,"dst")
                        deviceName = getDeviceName(macAddr)
                        macAddr = convertMac(macAddr)
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
                        #print macAddr
                        #print "device"+deviceName
                        # Check for deviceName before write to database.
                        if deviceName == "NONE":
                            print "SKIP!"
                        else:
                            pushToDB(ipAddr,macAddr,portSrc,portDst,pktLength,deviceName,protocolID,detail)
                    else:
                        pass
                except:
                    print ":("
            cap.close()
    #except:
    #    print "error"
    #    pass


def pushToDB(ipAddr,macAddr,portSrc,portDst,pktLength,deviceName,protocolID,detail):
    try:
        sqliteConnection = sqlite3.connect(TRAIN_PAPER)
        insertToTable = "INSERT INTO RESULT(ipAddr,macAddr,portSrc,portDst,pktLength,deviceName,protocol,detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        dataTuple = (ipAddr,macAddr,portSrc,portDst,pktLength,deviceName,protocolID,detail)
        cursor = sqliteConnection.cursor()
        cursor.execute(insertToTable, dataTuple)
        sqliteConnection.commit()
        print(ipAddr + ", " + macAddr + ", " + portSrc + ", " + portDst + ", " + pktLength + ", " + deviceName + ", "  + protocolID)
        cursor.close()
    except:
        print "DUPLICATE DATA :("
        pass
    finally:
        closeConnection(sqliteConnection)

def getOUI(macAddr):
    sqliteConnection = sqlite3.connect(MAC_OUI)
    cursor = sqliteConnection.cursor()
    executeCommand = "SELECT companyName FROM MAC_VENDORS WHERE oui = ?"
    cursor.execute(executeCommand, (macAddr,))
    result = cursor.fetchall()
    cursor.close()
    if result == []:
        return("[]")
    else:
        return str(result[0])

def getMac(currentCap,protocolChk,srcOrDst):
    subsEthernet = "eth"
    subsWLAN = "wlan"
    if subsEthernet in protocolChk and srcOrDst == "src":
        return str(currentCap.eth.src)
    elif subsEthernet in protocolChk and srcOrDst == "dst":
        return str(currentCap.eth.dst)
    elif subsWLAN in protocolChk and srcOrDst == "src":
        return str(currentCap.wlan.sa)
    elif subsWLAN in protocolChk and srcOrDst == "dst":
        return str(currentCap.wlan.da)
    else:
        pass
    return macAddr

def convertMac(macAddr):
    macAddr = macAddr.upper()
    macAddrP1 = macAddr[0:13]
    oui = getOUI(macAddrP1)
    if oui == "[]":
        macAddrP2 = macAddr[0:10]
        oui = getOUI(macAddrP2)
        if oui == "[]":
            macAddrP3 = macAddr[0:8]
            oui = getOUI(macAddrP3)
    oui = oui.replace("(u'", "").replace("',)", "") # Remove tuple symbol > ( ,)
    return oui

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

def getIP(currentCap):
    ip192 = "ip192."
    ip10 = "ip10."
    srcIP = "ip" + str(currentCap.ip.src)
    dstIP = "ip" + str(currentCap.ip.dst)
    # If we need destination ip address, then we need source mac address.
    if ip192 in srcIP or ip10 in srcIP:
        return str(currentCap.ip.dst),"src"
    else:
        return str(currentCap.ip.src),"dst"

def getDeviceName(macAddr):
    if "cc:50:e3:da:00:3f" in macAddr:
        deviceName = "0" # Anitech
    elif "60:01:94:ac:8f:fd" in macAddr:
        deviceName = "1" # Sonoff B1
    elif "60:01:94:74:22:a6" in macAddr:
        deviceName = "1" # Sonoff S20
    elif "84:f3:eb:3d:fa:f5" in macAddr:
        deviceName = "1" # Sonoff S26
    elif "50:c7:bf:8d:87:b6" in macAddr:
        deviceName = "2" # TP-Link LB100
    elif "b0:4e:26:ae:47:e5" in macAddr:
        deviceName = "2" # TP-Link HS110
    elif "ac:84:c6:bf:fc:a5" in macAddr:
        deviceName = "3" # TP-Link KC120
    elif "00:17:88:b2:6b:0c" in macAddr:
        deviceName = "4" # Philips Hue Bridge
    elif "18:b4:30:8f:88:a8" in macAddr:
        deviceName = "5" # Google Nest Cam IQ
    elif "54:e5:bd:8c:5c:5e" in macAddr:
        deviceName = "6" # GuRobot
    elif "68:9a:87:31:d8:15" in macAddr:
        deviceName = "7" # Amazon Echo
    elif "44:65:0d:56:cc:d3" in macAddr:
        deviceName = "7" # K.Arunan's Amazon Echo
    elif "4c:17:44:df:1f:b6" in macAddr:
        deviceName = "7" # Amazon Echo Spot
    else:
        deviceName = "NONE"
    return deviceName

def fileList():
    fileInDirectory = glob.glob("/home/scottishfold/Desktop/Database/TEST_PCAP/*.pcapng")
    return fileInDirectory
# Run
#createDatabase()

main()
