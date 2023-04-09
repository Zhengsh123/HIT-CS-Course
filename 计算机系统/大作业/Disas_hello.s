
hello.o：     文件格式 elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
   0:	55                   	push   %rbp
   1:	48 89 e5             	mov    %rsp,%rbp
   4:	48 83 ec 20          	sub    $0x20,%rsp
   8:	89 7d ec             	mov    %edi,-0x14(%rbp)
   b:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
   f:	83 7d ec 03          	cmpl   $0x3,-0x14(%rbp)
  13:	74 16                	je     2b <main+0x2b>
  15:	48 8d 3d 00 00 00 00 	lea    0x0(%rip),%rdi        # 1c <main+0x1c>
			18: R_X86_64_PC32	.rodata-0x4
  1c:	e8 00 00 00 00       	callq  21 <main+0x21>
			1d: R_X86_64_PLT32	puts-0x4
  21:	bf 01 00 00 00       	mov    $0x1,%edi
  26:	e8 00 00 00 00       	callq  2b <main+0x2b>
			27: R_X86_64_PLT32	exit-0x4
  2b:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  32:	eb 3b                	jmp    6f <main+0x6f>
  34:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  38:	48 83 c0 10          	add    $0x10,%rax
  3c:	48 8b 10             	mov    (%rax),%rdx
  3f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  43:	48 83 c0 08          	add    $0x8,%rax
  47:	48 8b 00             	mov    (%rax),%rax
  4a:	48 89 c6             	mov    %rax,%rsi
  4d:	48 8d 3d 00 00 00 00 	lea    0x0(%rip),%rdi        # 54 <main+0x54>
			50: R_X86_64_PC32	.rodata+0x21
  54:	b8 00 00 00 00       	mov    $0x0,%eax
  59:	e8 00 00 00 00       	callq  5e <main+0x5e>
			5a: R_X86_64_PLT32	printf-0x4
  5e:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 64 <main+0x64>
			60: R_X86_64_PC32	sleepsecs-0x4
  64:	89 c7                	mov    %eax,%edi
  66:	e8 00 00 00 00       	callq  6b <main+0x6b>
			67: R_X86_64_PLT32	sleep-0x4
  6b:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
  6f:	83 7d fc 09          	cmpl   $0x9,-0x4(%rbp)
  73:	7e bf                	jle    34 <main+0x34>
  75:	e8 00 00 00 00       	callq  7a <main+0x7a>
			76: R_X86_64_PLT32	getchar-0x4
  7a:	b8 00 00 00 00       	mov    $0x0,%eax
  7f:	c9                   	leaveq 
  80:	c3                   	retq   
