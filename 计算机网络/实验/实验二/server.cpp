#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib,"ws2_32.lib")
#define SERVER_PORT 12340 // 端口号
#define SERVER_IP "0.0.0.0" // IP 地址
#define SEQ_SIZE 16 // 序列号个数
#define SWIN_SIZE 8 // 发送窗口大小
#define RWIN_SIZE 8 // 接收窗口大小
#define BUFFER_SIZE 1024 // 缓冲区大小
#define LOSS_RATE 0.8 //丢包率
using namespace std;
struct recv {
	bool used;
	char buffer[BUFFER_SIZE];
	recv() {
		used = false;
		ZeroMemory(buffer, sizeof(buffer));
	}
}recvWindow[SEQ_SIZE];
struct send {
	clock_t start;//由于使用的是SR，因此每一个窗口位置都需要设置一个计时器
	char buffer[BUFFER_SIZE];
	send() {
		start = 0;
		ZeroMemory(buffer, sizeof(buffer));
	}
}sendWindow[SEQ_SIZE];
char cmdBuffer[50];
char buffer[BUFFER_SIZE];
char cmd[10];
char fileName[40];
char filePath[50];
char file[1024 * 1024];
int len = sizeof(SOCKADDR);
int recvSize;
int Deliver(char* file, int ack);
int Send(ifstream& infile, int seq, SOCKET socket, SOCKADDR* addr);
int MoveSendWindow(int seq);
int Read(ifstream& infile, char* buffer);
//主函数
int main(int argc, char* argv[]) {
	// 加载套接字库
	WORD wVersionRequested;
	WSADATA wsaData;
	// 版本 2.2
	wVersionRequested = MAKEWORD(2, 2);
	int err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) {
		printf("Winsock.dll 加载失败，错误码: %d\n", err);
		return -1;
	}
	if (LOBYTE(wsaData.wVersion) != LOBYTE(wVersionRequested) || HIBYTE(wsaData.wVersion) != HIBYTE(wVersionRequested)) {
		printf("找不到 %d.%d 版本的 Winsock.dll\n", LOBYTE(wVersionRequested), HIBYTE(wVersionRequested));
		WSACleanup();
		return -1;
	}
	else {
		printf("Winsock %d.%d 加载成功\n", LOBYTE(wVersionRequested), HIBYTE(wVersionRequested));
	}
	// 创建服务器套接字
	SOCKET socketServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// 设置为非阻塞模式
	int iMode = 1;
	ioctlsocket(socketServer, FIONBIO, (u_long FAR*) & iMode);
	SOCKADDR_IN addrServer;
	inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(SERVER_PORT);
	// 绑定端口
	if (err = bind(socketServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR))) {
		err = GetLastError();
		printf("绑定端口 %d 失败，错误码: % d\n", SERVER_PORT, err);
		WSACleanup();
		return -1;
	}
	else {
		printf("绑定端口 %d 成功", SERVER_PORT);
	}
	SOCKADDR_IN addrClient;
	int status = 0;
	clock_t start;
	clock_t now;
	int seq;
	int ack;
	ofstream outfile;
	ifstream infile;
	//进入接收状态，注意服务器主要处理的任务是接收客户机请求，共有上载和下载两种任务
	while (true) {
		recvSize = recvfrom(socketServer, buffer, BUFFER_SIZE, 0, ((SOCKADDR*)&addrClient), &len);
		if ((float)rand() / RAND_MAX > LOSS_RATE) {
			recvSize = 0;
			buffer[0] = 0;
		}
		switch (status)
		{
		case 0://接收请求
			if (recvSize > 0 && buffer[0] == 10) {
				char addr[100];
				ZeroMemory(addr, sizeof(addr));
				inet_ntop(AF_INET, &addrClient.sin_addr, addr, sizeof(addr));
				sscanf_s(buffer + 1, "%s%s", cmd, sizeof(cmd) - 1, fileName, sizeof(fileName) - 1);
				if (strcmp(cmd, "upload") && strcmp(cmd, "download")) {
					continue;
				}
				strcpy_s(filePath, "./");
				strcat_s(filePath, fileName);
				printf("收到来自客户端 %s 的请求: %s\n", addr, buffer);
				printf("是否同意该请求(Y/N)?");
				gets_s(cmdBuffer, 50);
				if (!strcmp(cmdBuffer, "Y")) {
					buffer[0] = 100;
					strcpy_s(buffer + 1, 3, "OK");
					if (!strcmp(cmd, "upload")) {
						file[0] = 0;
						start = clock();
						ack = 0;
						status = 1;
						outfile.open(filePath);
					}
					else if (!strcmp(cmd, "download")) {
						start = clock();
						seq = 0;
						status = -1;
						infile.open(filePath);
					}
				}
				else {
					buffer[0] = 100;
					strcpy_s(buffer + 1, 3, "NO");
				}
				sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
			}
			break;
		case 1://客户机请求上传，也就是服务器端是接收方
			if (recvSize > 0) {
				if (buffer[0] == 10) {
					if (!strcmp(buffer + 1, "Finish")) {
						printf("传输完毕...\n");
						start = clock();
						sendWindow[0].start = start - 1000L;
						sendWindow[0].buffer[0] = 100;
						strcpy_s(sendWindow[0].buffer + 1, 3, "OK");
						outfile.write(file, strlen(file));
						status = 2;
					}
					buffer[0] = 100;
					strcpy_s(buffer + 1, 3, "OK");
					sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
				}
				else if (buffer[0] == 20) {
					seq = buffer[1];
					int temp = seq - 1 - ack;
					if (temp < 0) {
						temp += SEQ_SIZE;
					}
					start = clock();
					seq--;
					if (temp < RWIN_SIZE) {
						if (!recvWindow[seq].used) {
							recvWindow[seq].used = true;
							strcpy_s(recvWindow[seq].buffer, strlen(buffer + 2) + 1, buffer + 2);
						}
						if (ack == seq) {
							ack = Deliver(file, ack);
						}
					}
					printf("接收数据帧 seq = %d, data = %s, 发送 ack = %d, 起始 ack = %d\n", seq + 1, buffer + 2, seq + 1, ack + 1);
					buffer[0] = 101;
					buffer[1] = seq + 1;
					buffer[2] = 0;
					sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
				}
			}
			break;
		case 2://接收完成
			if (recvSize > 0 && buffer[0] == 10 && !strcmp(buffer + 1, "OK")) {
				printf("传输成功，结束通信\n");
				status = 0;
				outfile.close();
			}
			now = clock();
			if (now - sendWindow[0].start >= 1000L) {
				sendWindow[0].start = now;
				sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
			}
			break;
		case -1://客户机请求下载，也就是服务器端充当发送方
			if (recvSize > 0) {
				if (buffer[0] == 10) {
					if (!strcmp(buffer + 1, "OK")) {
						printf("开始传输...\n");
						start = clock();
						status = -2;
					}
					buffer[0] = 100;
					strcpy_s(buffer + 1, 3, "OK");
					sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
				}
			}
			break;
		case -2://服务器端发送数据
			if (recvSize > 0 && buffer[0] == 11) {
				start = clock();
				ack = buffer[1];
				ack--;
				sendWindow[ack].start = -1L;
				if (ack == seq) {
					seq = MoveSendWindow(seq);
				}
				printf("接收 ack = %d, 当前起始 seq = %d\n", ack + 1, seq + 1);
			}
			if (!Send(infile, seq, socketServer, (SOCKADDR*)&addrClient)) {
				printf("传输完毕...\n");
				status = -3;
				start = clock();
				sendWindow[0].buffer[0] = 100;
				strcpy_s(sendWindow[0].buffer + 1, 7, "Finish");
				sendWindow[0].start = start - 1000L;
			}
			break;
		case -3://请求完成
			if (recvSize > 0 && buffer[0] == 10) {
				if (!strcmp(buffer + 1, "OK")) {
					printf("传输成功，结束通信\n");
					infile.close();
					status = 0;
					break;
				}
			}
			now = clock();
			if (now - sendWindow[0].start >= 1000L) {
				sendWindow[0].start = now;
				sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
			}
		default:
			break;
		}
		if (status != 0 && clock() - start > 5000L) {
			printf("通信超时, 结束通信\n");
			status = 0;
			outfile.close();
			continue;
		}
		if (recvSize <= 0) {
			Sleep(20);
		}
	}
	//关闭套接字，卸载库
	closesocket(socketServer);
	WSACleanup();
	return 0;
}
int Read(ifstream& infile, char* buffer) {
	//从文件中读取需要发送的数据
	if (infile.eof()) {
		return 0;
	}
	infile.read(buffer, 3);
	int cnt = infile.gcount();
	buffer[cnt] = 0;
	return cnt;
}
int Deliver(char* file, int ack) {
	while (recvWindow[ack].used) {
		recvWindow[ack].used = false;
		strcat_s(file, strlen(file) + strlen(recvWindow[ack].buffer) + 1, recvWindow[ack].buffer);
		ack++;
		ack %= SEQ_SIZE;
	}
	return ack;
}
int Send(ifstream& infile, int seq, SOCKET socket, SOCKADDR* addr) {
	//发送数据
	clock_t now = clock();
	for (int i = 0; i < SWIN_SIZE; i++) {
		int j = (seq + i) % SEQ_SIZE;
		if (sendWindow[j].start == -1L) {//传输超时，不需要
			continue;
		}
		if (sendWindow[j].start == 0L) {//开始计时
			if (Read(infile, sendWindow[j].buffer + 2)) {
				sendWindow[j].start = now;
				sendWindow[j].buffer[0] = 200;
				sendWindow[j].buffer[1] = j + 1;
			}
			else if (i == 0) {
				return 0;
			}
			else {
				break;
			}
		}
		else if (now - sendWindow[j].start >= 1000L) {//更新时间
			sendWindow[j].start = now;
		}
		else {
			continue;
		}
		printf("发送数据帧 seq = %d, data = %s\n", j + 1, sendWindow[j].buffer + 2);
		sendto(socket, sendWindow[j].buffer, strlen(sendWindow[j].buffer) + 1, 0, addr, sizeof(SOCKADDR));
	}
	return 1;
}

int MoveSendWindow(int seq) {
	//移动窗口
	while (sendWindow[seq].start == -1L) {
		sendWindow[seq].start = 0L;
		seq++;
		seq %= SEQ_SIZE;
	}
	return seq;
}