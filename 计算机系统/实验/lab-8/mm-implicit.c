/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 *
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
 
#include "mm.h"
#include "memlib.h"
 
/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "1190300321",
    /* First member's full name */
    "Zheng Shenghe",
    /* First member's email address */
    "531905990@qq.com",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""
};
/* Basic constants and macros */

#define WSIZE     4
#define DSIZE     8
#define INITCHUNKSIZE (1<<6)
#define CHUNKSIZE (1<<12)
#define MAX_LEN     16
#define ALIGN(size) ((((size) + (DSIZE-1)) / (DSIZE)) * (DSIZE)) //对齐

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
 
#define PACK(size, alloc) ((size) | (alloc))
 
/* Read and write a word at address p */
#define GET(p)       (*(size_t *)(p))
#define PUT(p, val)  (*(size_t *)(p) = (val))
 
/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)
 
/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp)       ((char *)(bp) - WSIZE)
#define FTRP(bp)       ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)
 
/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)  ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp)  ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))
 
#define SET_PTR(p, bp) (*(unsigned int *)(p) = (unsigned int)(bp))
 
#define PRED_PTR(bp) ((char *)(bp))
#define SUCC_PTR(bp) ((char *)(bp) + WSIZE)
 
#define PRED(bp) (*(char **)(bp))
#define SUCC(bp) (*(char **)(SUCC_PTR(bp)))
 
/* Global variables */
static char *heap_listp;  /* pointer to first block */ 
static void *extend_heap(size_t size);
static void *coalesce(void *bp);
static void *place(void *bp, size_t size);
static void printblock(void *bp);
static void checkblock(void *bp);
static void InsertNode(void *bp, size_t size); //插入到空闲链表
static void DeleteNode(void *bp);  //删除
void *Lists[MAX_LEN];  //分离空闲链表 
 
/*初始化内存分配器*/
int mm_init(void)
{
    int i;
    /* 初始化分离空闲链表 */
    for (i = 0; i < MAX_LEN; i++)
    {
        Lists[i] = NULL;
    }
    if ((heap_listp = mem_sbrk(4*WSIZE)) == NULL)
        return -1;
    PUT(heap_listp, 0);   //对齐填充
    PUT(heap_listp + (1 * WSIZE), PACK(DSIZE, 1));  //序言块
    PUT(heap_listp + (2 * WSIZE), PACK(DSIZE, 1));
    PUT(heap_listp + (3 * WSIZE), PACK(0, 1));  //结尾块
 
    /* Extend the empty heap with a free block of INITCHUNKSIZE bytes */
    if (extend_heap(INITCHUNKSIZE) == NULL)
    return -1;
    return 0;
}
 
/*mm_free - Free a block*/
void mm_free(void *bp)
{
    size_t size = GET_SIZE(HDRP(bp));
 
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    InsertNode(bp, size);
    coalesce(bp);
}
 
/*扩展堆*/
static void *extend_heap(size_t size)
{
    char *bp;
    /* Allocate an even number of words to maintain alignment */
    size = ALIGN(size);
    if ((bp = mem_sbrk(size)) == (void *)-1)
        return NULL;
    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));
    /*插入到分离空闲表中 */
    InsertNode(bp, size);
    /* Coalesce if the previous block was free */
    return coalesce(bp);
}
 
/*插入到空闲链表*/
static void InsertNode(void *bp, size_t size)
{
    int i = 0;
    void *search_bp = NULL;
    void *insert_bp = NULL;
 
    while((i<MAX_LEN-1)&&(size>1))  // 根据size的大小找到对应的分离空闲链表
    {
        size >>= 1;
        i++;
    }
    /* 找到分离空闲链表，在该链中寻找对应的插入位置，并且保持链中块由小到大分布 */
    search_bp = Lists[i];
    while ((search_bp != NULL) && (size > GET_SIZE(HDRP(search_bp))))
    {
        insert_bp = search_bp;
        search_bp = PRED(search_bp);
    }
 
    /* 四种情况 */
    if (search_bp != NULL)
    {
        /* 在中间插入*/
        if (insert_bp != NULL)
        {
            SET_PTR(PRED_PTR(bp), search_bp);
            SET_PTR(SUCC_PTR(search_bp), bp);
            SET_PTR(SUCC_PTR(bp), insert_bp);
            SET_PTR(PRED_PTR(insert_bp), bp);
        }
        /* 2. 在开头插入*/
        else
        {
            SET_PTR(PRED_PTR(bp), search_bp);
            SET_PTR(SUCC_PTR(search_bp), bp);
            SET_PTR(SUCC_PTR(bp), NULL);
            Lists[i] = bp;
        }
    }
    else
    {
        if (insert_bp != NULL)
        { /*在结尾插入*/
            SET_PTR(PRED_PTR(bp), NULL);
            SET_PTR(SUCC_PTR(bp), insert_bp);
            SET_PTR(PRED_PTR(insert_bp), bp);
        }
        else
        { /*第一次插入 */
            SET_PTR(PRED_PTR(bp), NULL);
            SET_PTR(SUCC_PTR(bp), NULL);
            Lists[i] = bp;
        }
    }
}
 
/*从空闲链表中删除*/
static void DeleteNode(void *bp)
{
    int i = 0;
    size_t size = GET_SIZE(HDRP(bp));
 
    // 根据size的大小找到对应的分离空闲链表
    while ((i < MAX_LEN - 1) && (size > 1))
    {
        size >>= 1;
        i++;
    }
    /* 四种可能性 */
    if (PRED(bp) != NULL)
    {
        /* 中间删除 */
        if (SUCC(bp) != NULL)
        {
            SET_PTR(SUCC_PTR(PRED(bp)), SUCC(bp));
            SET_PTR(PRED_PTR(SUCC(bp)), PRED(bp));
        }
        /* 表头删除，后面有块*/
        else
        {
            SET_PTR(SUCC_PTR(PRED(bp)), NULL);
            Lists[i] = PRED(bp);
        }
    }
    else
    {
        /* 3. 结尾删除*/
        if (SUCC(bp) != NULL)
        {
            SET_PTR(PRED_PTR(SUCC(bp)), NULL);
        }
        /* 4. 第一次删除*/
        else
        {
            Lists[i] = NULL;
        }
    }
}
 
/*合并块*/
static void *coalesce(void *bp)
{
    size_t  prev_alloc = GET_ALLOC(HDRP(PREV_BLKP(bp)));
    size_t  next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t size = GET_SIZE(HDRP(bp));
    /*四种情况*/
    if (prev_alloc && next_alloc)   /*case1*/
    {
        return bp;
    }
 
    else if (prev_alloc && !next_alloc)   /*case2*/
    {
        DeleteNode(bp);
        DeleteNode(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
    }
 
    else if (!prev_alloc && next_alloc)   /*case3*/
    {
        DeleteNode(bp);
        DeleteNode(PREV_BLKP(bp));
        size += GET_SIZE(HDRP(PREV_BLKP(bp)));
        PUT(FTRP(bp), PACK(size, 0));
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
    }
 
    else   /*case4*/
    {
        DeleteNode(bp);
        DeleteNode(PREV_BLKP(bp));
        DeleteNode(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(HDRP(NEXT_BLKP(bp)));
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
    }
 /* 合并后的free块插入到空闲链接表中 */
    InsertNode(bp, size);
 
    return bp;
}
 
/*分配块*/
void *mm_malloc(size_t size)
{
    char *bp = NULL;
    int i = 0;
    if (size == 0)
        return NULL;
 
    if (size <= DSIZE)
        size = 2*DSIZE;
    else
        size = ALIGN(size+DSIZE);  //内存对齐
 
    size_t asize = size;
 
    while (i < MAX_LEN)
    {
        /* 先找合适的空闲链表 */
        if (((asize <= 1) && (Lists[i] != NULL)))
        {
            bp = Lists[i];
            /* 找到链表，在该链寻找大小合适的未分配块 */
            while ((bp != NULL) && ((size > GET_SIZE(HDRP(bp)))))
                bp = PRED(bp);
 
            /* 找到对应的未分配的块 */
            if (bp != NULL)
                break;
        }
        asize >>= 1;
        i++;
    }
 
    /* 没有找到合适的未分配块，则扩展堆 */
    if (bp == NULL){
        if ((bp = extend_heap(MAX(size, CHUNKSIZE))) == NULL)
            return NULL;
    }
    /* 在未分配块中allocate size大小的块 */
    bp = place(bp, size);
 
    return bp;
}
 
/*将大小字节的块放在空闲块bp的开始处，并且如果余数至少是最小块大小则拆分*/
static void *place(void *bp, size_t asize)
{
    size_t csize = GET_SIZE(HDRP(bp));
    size_t remaining = csize - asize; /* allocate size大小的空间后剩余的大小 */
 
    DeleteNode(bp);
 
    /* 如果剩余的大小小于最小块，则不分离原块 */
    if (remaining < DSIZE * 2)
    {
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
 
    else if (asize >= 96)
    {
        PUT(HDRP(bp), PACK(remaining, 0));
        PUT(FTRP(bp), PACK(remaining, 0));
        PUT(HDRP(NEXT_BLKP(bp)), PACK(asize, 1));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(asize, 1));
        InsertNode(bp, remaining);
        return NEXT_BLKP(bp);
    }
 
    else
    {
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        PUT(HDRP(NEXT_BLKP(bp)), PACK(remaining, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(remaining, 0));
        InsertNode(NEXT_BLKP(bp), remaining);
    }
    return bp;
}
 
/*改进后的重新分配函数*/
void *mm_realloc(void *bp, size_t size)
{
    void *new_p = bp;
    int remaining;
    /*Ingore spurious requests*/
    if (size == 0)
        return NULL;
 
    if (size <= DSIZE)
        size = 2 * DSIZE;
    else
        size = ALIGN(size + DSIZE);  //内存对齐
 
    /* 如果size小于原来块的大小，直接返回原来的块 */
    if ((remaining = GET_SIZE(HDRP(bp)) - size) >= 0)
        return bp;
 
    /* 否则先检查地址连续下一个块是否为未分配块或者该块是堆的结束块 */
    else if (!GET_ALLOC(HDRP(NEXT_BLKP(bp))) || !GET_SIZE(HDRP(NEXT_BLKP(bp))))
    {
        /* 如果加上后面连续地址上的未分配块空间也不够，那么需要扩展块 */
        if ((remaining =GET_SIZE(HDRP(bp))+GET_SIZE(HDRP(NEXT_BLKP(bp)))-size)<0)
        {
            if (extend_heap(MAX(-remaining, CHUNKSIZE)) == NULL)
                return NULL;
            remaining +=MAX(-remaining,CHUNKSIZE);
        }
 
        /* 从分离空闲链表中删除刚刚利用的未分配块并设置新块的头尾 */
        DeleteNode(NEXT_BLKP(bp));
        PUT(HDRP(bp), PACK(size + remaining, 1));
        PUT(FTRP(bp), PACK(size + remaining, 1));
    }
    /* 如果没有可以利用的连续未分配块，只能申请新的不连续的未分配块 */
    else
    {
        new_p = mm_malloc(size);
        memcpy(new_p, bp, GET_SIZE(HDRP(bp)));
        mm_free(bp);
    }
    return new_p;
}
 
/*检查堆的一致性*/
void mm_checkheap(int verbose)
{
    char *bp = heap_listp;
 
    if (verbose)
	printf("Heap (%p):\n", heap_listp);
 
    if ((GET_SIZE(HDRP(heap_listp)) != DSIZE) || !GET_ALLOC(HDRP(heap_listp)))
	printf("Bad prologue header\n");
    checkblock(heap_listp);
 
    for (bp = heap_listp; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
	if (verbose)
	    printblock(bp);
	checkblock(bp);
    }
 
    if (verbose)
	printblock(bp);
    if ((GET_SIZE(HDRP(bp)) != 0) || !(GET_ALLOC(HDRP(bp))))
	printf("Bad epilogue header\n");
}
 
static void printblock(void *bp)
{
    size_t hsize, halloc, fsize, falloc;
 
    hsize = GET_SIZE(HDRP(bp));
    halloc = GET_ALLOC(HDRP(bp));
    fsize = GET_SIZE(FTRP(bp));
    falloc = GET_ALLOC(FTRP(bp));
 
    if (hsize == 0) {
	printf("%p: EOL\n", bp);
	return;
    }
 
    printf("%p: header: [%d:%c] footer: [%d:%c]\n", bp,
	   hsize, (halloc ? 'a' : 'f'),
	   fsize, (falloc ? 'a' : 'f'));
}
 
static void checkblock(void *bp)
{
    if ((size_t)bp % 8)
	printf("Error: %p is not doubleword aligned\n", bp);
    if (GET(HDRP(bp)) != GET(FTRP(bp)))
	printf("Error: header does not match footer\n");
}