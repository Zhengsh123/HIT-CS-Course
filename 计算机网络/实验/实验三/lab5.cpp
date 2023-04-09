/*
* THIS FILE IS FOR IP FORWARD TEST
*/
#include "sysInclude.h"
#include <vector>
using namespace std;
// system support
extern void fwd_LocalRcv(char *pBuffer, int length);

extern void fwd_SendtoLower(char *pBuffer, int length, unsigned int nexthop);

extern void fwd_DiscardPkt(char *pBuffer, int type);

extern unsigned int getIpv4Address();

// implemented by students
struct Route//路由表表项
{
    unsigned int mask;//掩码
    unsigned int dest;//目的地址
    unsigned int masklen;//掩码长度
    unsigned int nexthop;//下一跳地址

    Route(unsigned int dest,unsigned int masklen,unsigned int nexthop)
    {
        this->dest=ntohl(dest);
        this->masklen=ntohl(masklen);
        this->nexthop=ntohl(nexthop);
        this->mask=0;
        if(this->masklen)
        {
            this->mask = (int)0x80000000 >> (this->masklen - 1);
        }
    }
};

vector<Route> routeTable; // 路由表

/**
 * @brief 找路由表中是否有符合的表项
 * 
 * @param dstAddr 目的地址
 * @param nexthop 下一跳地址
 * @return bool 是否存在
 */
bool find_Next(unsigned dstAddr,unsigned int* nexthop)
{
    int size=routeTable.size();
    for(int i=0;i<size;i++)
    {
        Route route=routeTable[i];
        if((route.mask & dstAddr)==route.dest)
        {
            *nexthop= route.nexthop;
            return true;
        }
    }
    return false;
} 
/**
 * @brief 计算checksum
 * 
 * @param pBuffer IP报文头
 * @return unsigned short checksum值
 */
unsigned short cal_Checksum(char*pBuffer)
{
    unsigned int checkSum = 0;
	for (int i = 0; i < 10; i++)
	{
		checkSum += ((unsigned short*)pBuffer)[i];
	}
	checkSum = (checkSum >> 16) + (checkSum & 0xFFFF);
	checkSum = ~checkSum;
	return checkSum;
}
/**
 * @brief 初始化路由表(事实上本实验中就是清空路由表)
 * 
 */
void stud_Route_Init()
{
	routeTable.clear();
}
/**
 * @brief 路由表配置接口
 * 
 * @param proute 指向需要添加路由信息的结构体头部
 */
void stud_route_add(stud_route_msg *proute)
{
	Route route(proute->dest, proute->masklen, proute->nexthop);//构造路由项A
    routeTable.push_back(route);
}

/**
 * @brief 转发函数
 * 
 * @param pBuffer 指向接收到的 IPv4 分组头部
 * @param length IPv4 分组的长度
 * @return int 0 为成功， 1 为失败；
 */
int stud_fwd_deal(char *pBuffer, int length)
{
    unsigned char ttl=pBuffer[0];
    //ttl已经为0，
    if(ttl==0)
    {
        fwd_DiscardPkt(pBuffer, STUD_FORWARD_TEST_TTLERROR);
		return 1;
    }
    //如果是本机地址直接发送
    unsigned int dstAddr = ntohl(((unsigned int*)pBuffer)[4]);
    if(dstAddr==getIpv4Address())
    {
        fwd_LocalRcv(pBuffer,length);
        return 0;
    }
    //不是本机地址，找下一跳
    unsigned int* nexthop=new unsigned int;
    //表项中存在，处理
    if(find_Next(dstAddr, nexthop))
    {
       //注意要修改ttl和checksum
        pBuffer[8]--;//ttl-=1
        ((unsigned short *)pBuffer)[5]=0;
        ((unsigned short*)pBuffer)[5] = cal_Checksum(pBuffer);
		fwd_SendtoLower(pBuffer, length, *nexthop);
        delete nexthop;
        return 0;
    }
    delete nexthop;
    fwd_DiscardPkt(pBuffer, STUD_FORWARD_TEST_NOROUTE);
	return 1;
}

