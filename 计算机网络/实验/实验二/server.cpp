#include <stdlib.h>
#include <time.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <fstream>
#pragma comment(lib,"ws2_32.lib")
#define SERVER_PORT 12340 // �˿ں�
#define SERVER_IP "0.0.0.0" // IP ��ַ
#define SEQ_SIZE 16 // ���кŸ���
#define SWIN_SIZE 8 // ���ʹ��ڴ�С
#define RWIN_SIZE 8 // ���մ��ڴ�С
#define BUFFER_SIZE 1024 // ��������С
#define LOSS_RATE 0.8 //������
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
	clock_t start;//����ʹ�õ���SR�����ÿһ������λ�ö���Ҫ����һ����ʱ��
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
//������
int main(int argc, char* argv[]) {
	// �����׽��ֿ�
	WORD wVersionRequested;
	WSADATA wsaData;
	// �汾 2.2
	wVersionRequested = MAKEWORD(2, 2);
	int err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) {
		printf("Winsock.dll ����ʧ�ܣ�������: %d\n", err);
		return -1;
	}
	if (LOBYTE(wsaData.wVersion) != LOBYTE(wVersionRequested) || HIBYTE(wsaData.wVersion) != HIBYTE(wVersionRequested)) {
		printf("�Ҳ��� %d.%d �汾�� Winsock.dll\n", LOBYTE(wVersionRequested), HIBYTE(wVersionRequested));
		WSACleanup();
		return -1;
	}
	else {
		printf("Winsock %d.%d ���سɹ�\n", LOBYTE(wVersionRequested), HIBYTE(wVersionRequested));
	}
	// �����������׽���
	SOCKET socketServer = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// ����Ϊ������ģʽ
	int iMode = 1;
	ioctlsocket(socketServer, FIONBIO, (u_long FAR*) & iMode);
	SOCKADDR_IN addrServer;
	inet_pton(AF_INET, SERVER_IP, &addrServer.sin_addr);
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(SERVER_PORT);
	// �󶨶˿�
	if (err = bind(socketServer, (SOCKADDR*)&addrServer, sizeof(SOCKADDR))) {
		err = GetLastError();
		printf("�󶨶˿� %d ʧ�ܣ�������: % d\n", SERVER_PORT, err);
		WSACleanup();
		return -1;
	}
	else {
		printf("�󶨶˿� %d �ɹ�", SERVER_PORT);
	}
	SOCKADDR_IN addrClient;
	int status = 0;
	clock_t start;
	clock_t now;
	int seq;
	int ack;
	ofstream outfile;
	ifstream infile;
	//�������״̬��ע���������Ҫ����������ǽ��տͻ������󣬹������غ�������������
	while (true) {
		recvSize = recvfrom(socketServer, buffer, BUFFER_SIZE, 0, ((SOCKADDR*)&addrClient), &len);
		if ((float)rand() / RAND_MAX > LOSS_RATE) {
			recvSize = 0;
			buffer[0] = 0;
		}
		switch (status)
		{
		case 0://��������
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
				printf("�յ����Կͻ��� %s ������: %s\n", addr, buffer);
				printf("�Ƿ�ͬ�������(Y/N)?");
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
		case 1://�ͻ��������ϴ���Ҳ���Ƿ��������ǽ��շ�
			if (recvSize > 0) {
				if (buffer[0] == 10) {
					if (!strcmp(buffer + 1, "Finish")) {
						printf("�������...\n");
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
					printf("��������֡ seq = %d, data = %s, ���� ack = %d, ��ʼ ack = %d\n", seq + 1, buffer + 2, seq + 1, ack + 1);
					buffer[0] = 101;
					buffer[1] = seq + 1;
					buffer[2] = 0;
					sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
				}
			}
			break;
		case 2://�������
			if (recvSize > 0 && buffer[0] == 10 && !strcmp(buffer + 1, "OK")) {
				printf("����ɹ�������ͨ��\n");
				status = 0;
				outfile.close();
			}
			now = clock();
			if (now - sendWindow[0].start >= 1000L) {
				sendWindow[0].start = now;
				sendto(socketServer, sendWindow[0].buffer, strlen(sendWindow[0].buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
			}
			break;
		case -1://�ͻ����������أ�Ҳ���Ƿ������˳䵱���ͷ�
			if (recvSize > 0) {
				if (buffer[0] == 10) {
					if (!strcmp(buffer + 1, "OK")) {
						printf("��ʼ����...\n");
						start = clock();
						status = -2;
					}
					buffer[0] = 100;
					strcpy_s(buffer + 1, 3, "OK");
					sendto(socketServer, buffer, strlen(buffer) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
				}
			}
			break;
		case -2://�������˷�������
			if (recvSize > 0 && buffer[0] == 11) {
				start = clock();
				ack = buffer[1];
				ack--;
				sendWindow[ack].start = -1L;
				if (ack == seq) {
					seq = MoveSendWindow(seq);
				}
				printf("���� ack = %d, ��ǰ��ʼ seq = %d\n", ack + 1, seq + 1);
			}
			if (!Send(infile, seq, socketServer, (SOCKADDR*)&addrClient)) {
				printf("�������...\n");
				status = -3;
				start = clock();
				sendWindow[0].buffer[0] = 100;
				strcpy_s(sendWindow[0].buffer + 1, 7, "Finish");
				sendWindow[0].start = start - 1000L;
			}
			break;
		case -3://�������
			if (recvSize > 0 && buffer[0] == 10) {
				if (!strcmp(buffer + 1, "OK")) {
					printf("����ɹ�������ͨ��\n");
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
			printf("ͨ�ų�ʱ, ����ͨ��\n");
			status = 0;
			outfile.close();
			continue;
		}
		if (recvSize <= 0) {
			Sleep(20);
		}
	}
	//�ر��׽��֣�ж�ؿ�
	closesocket(socketServer);
	WSACleanup();
	return 0;
}
int Read(ifstream& infile, char* buffer) {
	//���ļ��ж�ȡ��Ҫ���͵�����
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
	//��������
	clock_t now = clock();
	for (int i = 0; i < SWIN_SIZE; i++) {
		int j = (seq + i) % SEQ_SIZE;
		if (sendWindow[j].start == -1L) {//���䳬ʱ������Ҫ
			continue;
		}
		if (sendWindow[j].start == 0L) {//��ʼ��ʱ
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
		else if (now - sendWindow[j].start >= 1000L) {//����ʱ��
			sendWindow[j].start = now;
		}
		else {
			continue;
		}
		printf("��������֡ seq = %d, data = %s\n", j + 1, sendWindow[j].buffer + 2);
		sendto(socket, sendWindow[j].buffer, strlen(sendWindow[j].buffer) + 1, 0, addr, sizeof(SOCKADDR));
	}
	return 1;
}

int MoveSendWindow(int seq) {
	//�ƶ�����
	while (sendWindow[seq].start == -1L) {
		sendWindow[seq].start = 0L;
		seq++;
		seq %= SEQ_SIZE;
	}
	return seq;
}