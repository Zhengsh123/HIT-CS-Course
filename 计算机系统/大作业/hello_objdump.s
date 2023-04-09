
hello：     文件格式 elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	f3 0f 1e fa          	endbr64 
  401004:	48 83 ec 08          	sub    $0x8,%rsp
  401008:	48 8b 05 e9 2f 00 00 	mov    0x2fe9(%rip),%rax        # 403ff8 <__gmon_start__>
  40100f:	48 85 c0             	test   %rax,%rax
  401012:	74 02                	je     401016 <_init+0x16>
  401014:	ff d0                	callq  *%rax
  401016:	48 83 c4 08          	add    $0x8,%rsp
  40101a:	c3                   	retq   

Disassembly of section .plt:

0000000000401020 <.plt>:
  401020:	ff 35 e2 2f 00 00    	pushq  0x2fe2(%rip)        # 404008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 e4 2f 00 00    	jmpq   *0x2fe4(%rip)        # 404010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <puts@plt>:
  401030:	ff 25 e2 2f 00 00    	jmpq   *0x2fe2(%rip)        # 404018 <puts@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	pushq  $0x0
  40103b:	e9 e0 ff ff ff       	jmpq   401020 <.plt>

0000000000401040 <printf@plt>:
  401040:	ff 25 da 2f 00 00    	jmpq   *0x2fda(%rip)        # 404020 <printf@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	pushq  $0x1
  40104b:	e9 d0 ff ff ff       	jmpq   401020 <.plt>

0000000000401050 <getchar@plt>:
  401050:	ff 25 d2 2f 00 00    	jmpq   *0x2fd2(%rip)        # 404028 <getchar@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	pushq  $0x2
  40105b:	e9 c0 ff ff ff       	jmpq   401020 <.plt>

0000000000401060 <exit@plt>:
  401060:	ff 25 ca 2f 00 00    	jmpq   *0x2fca(%rip)        # 404030 <exit@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	pushq  $0x3
  40106b:	e9 b0 ff ff ff       	jmpq   401020 <.plt>

0000000000401070 <sleep@plt>:
  401070:	ff 25 c2 2f 00 00    	jmpq   *0x2fc2(%rip)        # 404038 <sleep@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	pushq  $0x4
  40107b:	e9 a0 ff ff ff       	jmpq   401020 <.plt>

Disassembly of section .text:

0000000000401080 <_start>:
  401080:	f3 0f 1e fa          	endbr64 
  401084:	31 ed                	xor    %ebp,%ebp
  401086:	49 89 d1             	mov    %rdx,%r9
  401089:	5e                   	pop    %rsi
  40108a:	48 89 e2             	mov    %rsp,%rdx
  40108d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401091:	50                   	push   %rax
  401092:	54                   	push   %rsp
  401093:	49 c7 c0 b0 11 40 00 	mov    $0x4011b0,%r8
  40109a:	48 c7 c1 40 11 40 00 	mov    $0x401140,%rcx
  4010a1:	48 c7 c7 b5 10 40 00 	mov    $0x4010b5,%rdi
  4010a8:	ff 15 42 2f 00 00    	callq  *0x2f42(%rip)        # 403ff0 <__libc_start_main@GLIBC_2.2.5>
  4010ae:	f4                   	hlt    
  4010af:	90                   	nop

00000000004010b0 <_dl_relocate_static_pie>:
  4010b0:	f3 0f 1e fa          	endbr64 
  4010b4:	c3                   	retq   

00000000004010b5 <main>:
  4010b5:	55                   	push   %rbp
  4010b6:	48 89 e5             	mov    %rsp,%rbp
  4010b9:	48 83 ec 20          	sub    $0x20,%rsp
  4010bd:	89 7d ec             	mov    %edi,-0x14(%rbp)
  4010c0:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  4010c4:	83 7d ec 03          	cmpl   $0x3,-0x14(%rbp)
  4010c8:	74 16                	je     4010e0 <main+0x2b>
  4010ca:	48 8d 3d 37 0f 00 00 	lea    0xf37(%rip),%rdi        # 402008 <_IO_stdin_used+0x8>
  4010d1:	e8 5a ff ff ff       	callq  401030 <puts@plt>
  4010d6:	bf 01 00 00 00       	mov    $0x1,%edi
  4010db:	e8 80 ff ff ff       	callq  401060 <exit@plt>
  4010e0:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4010e7:	eb 3b                	jmp    401124 <main+0x6f>
  4010e9:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4010ed:	48 83 c0 10          	add    $0x10,%rax
  4010f1:	48 8b 10             	mov    (%rax),%rdx
  4010f4:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4010f8:	48 83 c0 08          	add    $0x8,%rax
  4010fc:	48 8b 00             	mov    (%rax),%rax
  4010ff:	48 89 c6             	mov    %rax,%rsi
  401102:	48 8d 3d 24 0f 00 00 	lea    0xf24(%rip),%rdi        # 40202d <_IO_stdin_used+0x2d>
  401109:	b8 00 00 00 00       	mov    $0x0,%eax
  40110e:	e8 2d ff ff ff       	callq  401040 <printf@plt>
  401113:	8b 05 2b 2f 00 00    	mov    0x2f2b(%rip),%eax        # 404044 <sleepsecs>
  401119:	89 c7                	mov    %eax,%edi
  40111b:	e8 50 ff ff ff       	callq  401070 <sleep@plt>
  401120:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
  401124:	83 7d fc 09          	cmpl   $0x9,-0x4(%rbp)
  401128:	7e bf                	jle    4010e9 <main+0x34>
  40112a:	e8 21 ff ff ff       	callq  401050 <getchar@plt>
  40112f:	b8 00 00 00 00       	mov    $0x0,%eax
  401134:	c9                   	leaveq 
  401135:	c3                   	retq   
  401136:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40113d:	00 00 00 

0000000000401140 <__libc_csu_init>:
  401140:	f3 0f 1e fa          	endbr64 
  401144:	41 57                	push   %r15
  401146:	4c 8d 3d 03 2d 00 00 	lea    0x2d03(%rip),%r15        # 403e50 <_DYNAMIC>
  40114d:	41 56                	push   %r14
  40114f:	49 89 d6             	mov    %rdx,%r14
  401152:	41 55                	push   %r13
  401154:	49 89 f5             	mov    %rsi,%r13
  401157:	41 54                	push   %r12
  401159:	41 89 fc             	mov    %edi,%r12d
  40115c:	55                   	push   %rbp
  40115d:	48 8d 2d ec 2c 00 00 	lea    0x2cec(%rip),%rbp        # 403e50 <_DYNAMIC>
  401164:	53                   	push   %rbx
  401165:	4c 29 fd             	sub    %r15,%rbp
  401168:	48 83 ec 08          	sub    $0x8,%rsp
  40116c:	e8 8f fe ff ff       	callq  401000 <_init>
  401171:	48 c1 fd 03          	sar    $0x3,%rbp
  401175:	74 1f                	je     401196 <__libc_csu_init+0x56>
  401177:	31 db                	xor    %ebx,%ebx
  401179:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401180:	4c 89 f2             	mov    %r14,%rdx
  401183:	4c 89 ee             	mov    %r13,%rsi
  401186:	44 89 e7             	mov    %r12d,%edi
  401189:	41 ff 14 df          	callq  *(%r15,%rbx,8)
  40118d:	48 83 c3 01          	add    $0x1,%rbx
  401191:	48 39 dd             	cmp    %rbx,%rbp
  401194:	75 ea                	jne    401180 <__libc_csu_init+0x40>
  401196:	48 83 c4 08          	add    $0x8,%rsp
  40119a:	5b                   	pop    %rbx
  40119b:	5d                   	pop    %rbp
  40119c:	41 5c                	pop    %r12
  40119e:	41 5d                	pop    %r13
  4011a0:	41 5e                	pop    %r14
  4011a2:	41 5f                	pop    %r15
  4011a4:	c3                   	retq   
  4011a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4011ac:	00 00 00 00 

00000000004011b0 <__libc_csu_fini>:
  4011b0:	f3 0f 1e fa          	endbr64 
  4011b4:	c3                   	retq   

Disassembly of section .fini:

00000000004011b8 <_fini>:
  4011b8:	f3 0f 1e fa          	endbr64 
  4011bc:	48 83 ec 08          	sub    $0x8,%rsp
  4011c0:	48 83 c4 08          	add    $0x8,%rsp
  4011c4:	c3                   	retq   
