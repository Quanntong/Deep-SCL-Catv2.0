<template>
  <div class="home">
    <h1>心理风险预测系统</h1>
    <el-row :gutter="20" style="margin-top: 40px;">
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/test')" style="cursor: pointer; height: 200px;">
          <template #header><el-icon><Edit /></el-icon> 在线心理测评</template>
          <p>完成 SCL-90 量表的 90 道题目，系统自动计算因子分并进行风险评估</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/manual')" style="cursor: pointer; height: 200px;">
          <template #header><el-icon><DataLine /></el-icon> 快速风险筛查</template>
          <p>已有因子分数据？直接输入 SCL-90 和 EPQ 因子分，快速获取风险评估结果</p>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" @click="$router.push('/batch')" style="cursor: pointer; height: 200px;">
          <template #header><el-icon><Upload /></el-icon> 批量数据分析</template>
          <p>上传 Excel 文件，批量分析多名学生的心理风险状况</p>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
</script>

<style scoped>
.home { padding: 40px; text-align: center; }
.home h1 { color: #409eff; }
</style>
