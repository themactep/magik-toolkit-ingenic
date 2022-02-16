#ifndef __IAAC_H__
#define __IAAC_H__

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif /* __cplusplus */

/**
 * Error Code:
 * IAAC [0x80000000 - 0xafffffff] soc(client) relative error code;
 * IAAS [0xb0000000 - 0xffffffff] server relative error code:
 *      0xb ==>process error except database relative
 *      0xc ==>database bussiness process error
 *      0xd ==>visit database process error
 *      0xe ==>modify database process error
 */
#define ERR_IAAC_INV_INFO           0x80000001      /**< IAACInfo参数错误 */
#define ERR_IAAC_INV_LICENSE_PATH   0x80000002      /**< License文件路径不可读写 */
#define ERR_IAAC_INV_SOCTYPE        0x80000003      /**< 不支持的芯片类型 */
#define ERR_IAAC_INV_SOCID          0x80000004      /**< 不支持的芯片标识 */
#define ERR_IAAC_INV_RWLOCK         0x80000005      /**< 创建读写锁错误 */

#define ERR_VERIFY_NO_LICENSE       0x90000001      /**< 不能得到license */
#define ERR_VERIFY_INV_LICENSE      0x90000002      /**< 错误的license */
#define ERR_VERIFY_ALGO_NOPAY       0x90000003      /**< 算法未付费 */
#define ERR_VERIFY_INV_ALGOLIB      0x90000004      /**< 算法库错误 */

/**
 * IAAC联网授权信息，默认不使用，只有在需要自己实现send_and_recv函数时使用
 */
typedef struct {
	char	        url[64];							/**< 远端服务的URL，需要调用程序解析成对应的IP */
	char            ip[64];						        /**< 远端服务的IP，若strlen(ip)的长度大于0，则可直接使用ip，否则使用url */
	int		        port;								/**< 远端服务的端口号 */
	unsigned char	need_send_data[128];	            /**< 需要发送到远端服务的数据 */
	int		        need_send_data_len;					/**< 需要发送到远端服务的实际数据长度 */
    unsigned char   *need_recv_data;                    /**< 向远端服务发送数据后，需要从远端服务接收的实际数据存放到此空间 */
	int		        need_recv_data_len;					/**< 向远端服务发送数据后，需要从远端服务接收的实际数据长度 */
} IAACAuthInfo;

/**
 * IAAC初始化信息，除过send_and_recv成员函数需考虑联网情况根据实际设置外，license_path和sn不能为空
 */
typedef struct {
	char    *license_path;  /*<< 许可证license文件路径名，以此文件路径名建立的文件必须允许可读可写，
                                每个算法占用72个字节，若一个产品上需要激活N个算法，则该文件大小最小为N*72个字节 */
	int     cid;		    /*<< 客户编号CID，由君正提供，客户必须保密以防信息流失，ADK2.0起该成员未使用 */
	int     fid;		    /*<< 功能编号FID，由君正提供，ADK2.0起该成员未使用 */
	char    *sn;            /*<< 激活序列号SN，最大32个字节，唯一不重复,可以由君正提供,
                                也可以由客户在产品生产前统一向君正提供,并由君正授权 */
    int     (*send_and_recv)(IAACAuthInfo *authInfo);   /**< 若君正芯片可连接外网，则置为NULL，否则你需要自己实现并赋值
                                可以与外网交互数据的send_and_recv接口函数，你必须根据authInfo里提供的信息将数据发送出去，
                                并接受need_recv_data_len大小的数据存到need_recv_data空间里 */
} IAACInfo;

/**
 * @fn int IAAC_Init(const IAACInfo *info)
 *
 * 初始化IAAC模块
 *
 * @param[in] info IAAC模块初始化信息结构体指针
 *
 * @retval 0 成功
 * @retval < 0 失败，返回错误码，错误码请转化为unsigned int并用0x%08x打印
 *
 * @remarks 此函数仅对IAAC做初始化，不做授权和激活，授权和激活统一在算法初始化函数里进行
 * @remarks 此函数最晚必须在每次启动程序后第一个需激活函数前调用
 * @remarks 请详细了解IAACInfo里成员函数send_and_recv，若产品能连接外网，IAAC里面会实现该函数，你只需要赋值为NULL即可，否则你需要自己
 * 按要求实现可以联外网并发送接收数据的函数并将其指针赋给该函数。
 * @remarks 若多线程或多进程调用算法初始化函数，则需要在这些算法初始化函数调用线程或进程创建前调用本函数IAAC_Init
 */
int IAAC_Init(IAACInfo *info);

/**
 * @fn int IAAC_DeInit(void)
 *
 * 反初始化IAAC模块
 *
 * @remarks 此函数仅对IAAC做反初始化，一旦开始反初始化，则之后不能执行需要授权和激活的算法初始化函数
 * @remarks 此函数最早必须在最后一个需要授权和激活的算法初始化函数调用完成之后才可调用
 */
void IAAC_DeInit(void);

/**
 * @fn int IAAC_SetAuthIntervalDays(int days)
 *
 * 设置算法授权天数
 *
 * @param[in] days 算法一次授权可激活的单位天数
 *
 * @retval 0 成功
 * @retval < 0 失败，返回错误码，错误码请转化为unsigned int并用0x%08x打印
 *
 * @remarks 若你需要自行设置算法一次授权可激活的单位天数，则需在需要激活的算法初始化函数前调用该函数设置，否则这个天数为商务过程获得的
 * 算法一次授权可激活的单位天数
 * @remarks 若需要调用该函数，则你需要保证该函数和其后需要激活的算法初始化函数作为一个运行单元，和别的需要激活的算法初始化串行执行；
 * @remarks 该函数和其后的需要激活的算法初始化函数执行后，days设置自动失效；故若需要对其他算法授权也设置一次授权可激活天数，则需要在其他
 * 算法前也调用该函数
 * @remarks 若已经授权的算法在有效期内，则调用该函数无效，否则使用该函数的参数days决定新的授权可激活的有效期；
 */
int IAAC_SetAuthIntervalDays(int days);

/**
 * @fn int IAAC_SetServerUrlIpAndPort(const char *url, const char *ip, int port)
 *
 * 设置算法授权服务器url或ip， 和端口
 *
 * @param[in] url 算法授权服务器url
 * @param[in] ip 算法授权服务器ip
 * @param[in] port 算法授权服务器端口
 *
 * @retval 0 成功
 * @retval < 0 失败，返回错误码，错误码请转化为unsigned int并用0x%08x打印
 *
 * @remarks 用于设置算法授权服务器相关信息，url或ip,和端口;
 * @remarks 必须紧跟在IAAC_Init之后调用;
 * @remarks 若设置ip，则使用ip，否则使用url;
 * @remarks 若不设置，默认为中国国内服务器的url;
 * @remarks 目前支持可设置的url和端口列表如下：
 * @remarks 美国：usaalive.sv.ingenic.com，端口：13283
 */
int IAAC_SetServerUrlIpAndPort(const char *url, const char *ip, int port);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __IAAC_H__ */
