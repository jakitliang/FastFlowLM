# FLM v1.0.13 Chatbox 上下文协议测试包

基线：ROCm/FastFlowLM 稳定标签 v1.0.13，提交 838cfca5d9370adb7cc5ba19746bed21e14ecd01。
这是修改后的测试构建，不是官方发行版。保持 XRT / AMD NPU 后端，未修改官方引擎 DLL、xclbin 或模型权重。

## 边界与安全

- 解压到新的独立目录，不覆盖已有 FLM。包内无模型、无驱动、无安装程序。
- 目标机器必须已有兼容 AMD NPU / XRT 驱动。构建使用官方版本标记 32.0.203.304；不在 Intel 构建机安装 AMD 驱动。
- 启动器默认使用回环地址和独立测试端口 52626；若端口被占用直接退出，不会停止现有服务。
- 不建议两个 FLM 服务同时占用同一个 NPU。实际推理测试前需由使用者协调现有服务，启动器不会擅自结束进程。
- 启动器需要 ModelHome 参数，指向包含 models 子目录的现有 FLM 数据目录，通常为当前用户的 .flm 目录。
- 启动器设置 FLM_NO_AUTO_PULL=1：聊天模型缺失或被新版本判为过期时拒绝加载，禁止自动下载或替换模型。
- 如 XRT DLL 不在系统搜索路径，可传 XrtRuntimeDir 指向现有运行库目录；不复制/安装驱动。
- 启动器只改自身进程环境，退出后恢复，不修改注册表或系统环境变量。Ctrl+C 结束测试服务。

## 协议

- /v1/models、/api/tags 和 /api/show 补充目录中的真实 details、capabilities、default_context_length。
- 这些列表仍是当前平台可用的**目录**，不是“全部已安装”的列表。
- 仅从本地已安装模型 config.json 的 max_position_embeddings（包括 text_config / thinker_config.text_config）读取模型声明上限。
- model_context_length 表示模型架构声明；context_capacity.max 同时受已注册后端的显式上限约束。不是承诺所有窗口都能在当前 NPU 内存中成功分配。
- 缺少真实最大值就不返回 context_capacity，不拿 default_context_length 充当最大容量。
- configured_context_length 是启动配置/目录默认值；不是当前已加载模型的窗口。当前实际窗口只看 /api/ps 的 context_length，包含后端实际 decode 上限。
- 未加载模型时 /api/ps 返回空 models 数组，不再返回 model-faker，也不伪造过期时间；OpenAI 模型列表不再伪造创建时间。
- OpenAI 聊天/补全支持顶层 num_ctx；原生 /api/chat、/api/generate 支持 options.num_ctx。二者都存在且冲突时拒绝。
- num_ctx 必须是大于等于 512 的整数，且不超过已知模型/后端上限。single-turn 固定窗口模型拒绝覆盖。
- 模型、后端或请求窗口改变会重新加载并清空旧 prompt cache；加载后核对实际窗口，不静默接受截断。
- 不传 num_ctx 或传 null 表示 Auto：回到启动 --ctx-len 配置，否则回到目录默认值。上一个请求的窗口不会粘到 Auto。
- max_tokens / max_completion_tokens / num_predict 仍是输出限制，不是上下文窗口。
- 为网页访问局域网地址时，可通过 AllowedOrigins 显式允许页面来源；不会默认放开为 *。

## 验证与已知限制

本机已执行：MSVC Windows x64 Release 编译/链接、Windows 原生元数据单元测试、现有 HTTP 守卫测试，以及 Linux CPU 元数据测试。
测试中的容量是独立的合成 fixture，不进入业务元数据。
本机为 Intel，未进行 AMD NPU 推理。分档切换、实际显存/内存分配、切回 Auto 和停止后的恢复，必须在目标 AMD NPU 上验证。
Chatbox 客户端还未启用此 FLM 分档协议；本包先验证服务端。

verify-http.py 只执行模型列表、/api/show 和 /api/ps 读取，不触发推理或下载。需要 Python 3 标准库，地址由参数显式提供。
生成请求仅应针对已经安装且与 v1.0.13 兼容的模型，在独立测试服务中人工确认后执行。

## 构建与来源

在 VS 2022 的 x64 Native Tools Command Prompt 中运行 build-windows.cmd，确保 CMake、Ninja 和 Rust 在 PATH 中。脚本使用 C++ 8 并行、Rust 4 并行，并显式启用 UTF-8 源码编码。
package.py 用同版本官方 Windows ZIP 的运行库/资源组包，只替换本机编译的主程序；会核对预编译引擎 DLL 与源码目录完全一致。
打包前先暂存或提交所有要随包分发的源码；不会收集未跟踪的本机脚本。source.patch 保存相对稳定基线的源码修改，source/ 保存对应文件；manifest.json 记录稳定提交、编译产物和官方 ZIP 的 SHA-256。
保留原始 LICENSE_RUNTIME.txt、LICENSE_BINARY.txt 与 THIRD_PARTY_NOTICES.txt。