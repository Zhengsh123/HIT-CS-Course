	.file	"hello.c"
	.text
	.globl	sleepsecs
	.data
	.align 4
	.type	sleepsecs, @object
	.size	sleepsecs, 4
sleepsecs:
	.long	2
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage: Hello 1190300321 \351\203\221\346\231\237\350\265\253\357\274\201"
.LC1:
	.string	"Hello %s %s\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	cmpl	$3, -20(%rbp)
	je	.L2
	leaq	.LC0(%rip), %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L2:
	movl	$0, -4(%rbp)
	jmp	.L3
.L4:
	movq	-32(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	sleepsecs(%rip), %eax
	movl	%eax, %edi
	call	sleep@PLT
	addl	$1, -4(%rbp)
.L3:
	cmpl	$9, -4(%rbp)
	jle	.L4
	call	getchar@PLT
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Debian 10.2.1-6) 10.2.1 20210110"
	.section	.note.GNU-stack,"",@progbits
