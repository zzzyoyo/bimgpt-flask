import grpc
import greet_pb2
import greet_pb2_grpc
import json


def snl(resps_str, dictionary):
    channel = grpc.insecure_channel('localhost:5175')  # 请使用实际的服务器地址和端口
    stub = greet_pb2_grpc.GreeterStub(channel)
    response = stub.GenerateSNL(greet_pb2.SNLRequest(resps=resps_str, dictionary=dictionary))
    results = json.loads(response.results)
    pass_rate = response.accepted / len(results)
    metrics = []
    return_results = []
    for result in results:
        return_results.append({
            "input": result["RootSegment"]["Input"],
            "output": result["RootSegment"]["Output"] if result["RootSegment"]["Output"] else "生成失败"
        })
        accepted_length = 0
        for accepted_sentence in result["Accepted"]:
            accepted_length += len(accepted_sentence["Input"])
        metrics.append(accepted_length / len(result["RootSegment"]["Input"]))
    metric = sum(metrics) / len(metrics)
    return_map = {
        "result": return_results,
        "metric": f'%.2f' % metric,
        "accepted": response.accepted,
        "warn": response.warn,
        "error": response.error,
        "pass_rate": f'%.2f' % pass_rate
    }
    print(return_map)
    return return_map
