/*
* THIS FILE IS FOR IP TEST
*/
// system support
#include "sysInclude.h"
#include<stdio.h>
#include<malloc.h>
extern void ip_DiscardPkt(char* pBuffer,int type);

extern void ip_SendtoLower(char*pBuffer,int length);

extern void ip_SendtoUp(char *pBuffer,int length);

extern unsigned int getIpv4Address();

// implemented by students
/**
 * @brief 判断是否能成功接收
 * 
 * @param pBuffer 指向接收缓冲区的指针，指向 IPv4 分组头部
 * @param length IPv4 分组长度
 * @return 0：成功接收 IP 分组并交给上层处理
           1： IP 分组接收失败 
 */
int stud_ip_recv(char *pBuffer,unsigned short length)
{
	int version=pBuffer[0]>>4;
  int headLength=pBuffer[0]&0xf;//取最后16位
  int TTL=(unsigned short)pBuffer[8];
  int checkSum=ntohs(*(unsigned short *)(pBuffer+10));
  int dstAddr=ntohl(*(unsigned int *)(pBuffer+16));
    
  //判断版本号是否出错
  if(version!=4)
  {
      ip_DiscardPkt(pBuffer, STUD_IP_TEST_VERSION_ERROR);
	    return 1;
  }
    //判断头部长度是否出错
  if(headLength<5)
  {
      ip_DiscardPkt(pBuffer, STUD_IP_TEST_HEADLEN_ERROR);
	return 1;
  }
    //判断TTL是否出错
  if(TTL<=0)
  {
    ip_DiscardPkt(pBuffer, STUD_IP_TEST_TTL_ERROR);
	  return 1;
  }
    //判断目的地址是否出错/本机或者广播
  if (dstAddr != getIpv4Address() && dstAddr != 0xffff){
	ip_DiscardPkt(pBuffer,STUD_IP_TEST_DESTINATION_ERROR);  
	return 1;
  }
  //判断校验和是否出错
  unsigned int sum = 0;
	for (int i = 0; i < 10; i++)
	{
		sum += ((unsigned short*)pBuffer)[i];
	}
	sum = (sum >> 16) + (sum & 0xFFFF);
	
	if (sum != 0xffff){
		ip_DiscardPkt(pBuffer, STUD_IP_TEST_CHECKSUM_ERROR);
		return 1;
	}
    //无错误
  ip_SendtoUp(pBuffer,length); 
	return 0;
}
/**
 * @brief 发送接口，封装IPV4数据报
 * 
 * @param pBuffer 指向发送缓冲区的指针，指向 IPv4 上层协议数据头部
 * @param len IPv4 上层协议数据长度
 * @param srcAddr 源 IPv4 地址
 * @param dstAddr 目的 IPv4 地址
 * @param protocol IPv4 上层协议号
 * @param ttl 生存时间（Time To Live）
 * @return 返回值：
            0：成功发送 IP 分组
            1：发送 IP 分组失败
 */
int stud_ip_Upsend(char *pBuffer,unsigned short len,unsigned int srcAddr,
				   unsigned int dstAddr,byte protocol,byte ttl)
{
    char *IPBuffer = (char *)malloc((20 + len) * sizeof(char));
    memset(IPBuffer,0,len+20);
    //版本号+头部长度
    IPBuffer[0]=0x45;
    //总长度
    unsigned short totalLength=htons(len+20);//转换字节序
    memcpy(IPBuffer+2,&totalLength,2);
    //TTL
    IPBuffer[8]=ttl;
    //协议
    IPBuffer[9]=protocol;
    //源地址与目的地址
    unsigned int src = htonl(srcAddr);  
	unsigned int dis = htonl(dstAddr);  
	memcpy(IPBuffer + 12, &src, 4);  //源与目的IP地址
	memcpy(IPBuffer + 16, &dis, 4); 

 
	//计算checksum
  unsigned int sum = 0;
  unsigned short checksum=0;
	for (int i = 0; i < 10; i++)
	{
		sum += ((unsigned short*)IPBuffer)[i];
	}
	sum = (sum >> 16) + (sum & 0xFFFF);

	checksum = ~sum; 

  memcpy(IPBuffer+10,&checksum ,2);
  memcpy(IPBuffer + 20, pBuffer, len);    
	ip_SendtoLower(IPBuffer,len+20);  
	return 0;
}
