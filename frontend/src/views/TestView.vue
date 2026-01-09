<template>
  <div class="test-view">
    <el-card>
      <template #header>
        <span>SCL-90 心理健康测评 (第 {{ currentPage }}/9 页)</span>
      </template>
      
      <el-progress :percentage="Math.round((currentPage / 9) * 100)" style="margin-bottom: 20px;" />
      
      <div v-for="(q, idx) in currentQuestions" :key="idx" class="question-item">
        <p>{{ q }}</p>
        <el-radio-group v-model="answers[getQuestionIndex(idx)]">
          <el-radio v-for="opt in options" :key="opt.value" :value="opt.value">{{ opt.label }}</el-radio>
        </el-radio-group>
      </div>
      
      <div style="margin-top: 20px; text-align: center;">
        <el-button @click="currentPage--" :disabled="currentPage === 1">上一页</el-button>
        <el-button v-if="currentPage < 9" type="primary" @click="currentPage++">下一页</el-button>
        <el-button v-else type="success" @click="submit" :loading="store.loading">提交评估</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { SCL90_QUESTIONS, ANSWER_OPTIONS } from '@/constants/questions'
import { usePredictionStore } from '@/stores/prediction'

const router = useRouter()
const store = usePredictionStore()
const currentPage = ref(1)
const answers = ref(Array(90).fill(1))
const options = ANSWER_OPTIONS

const currentQuestions = computed(() => {
  const start = (currentPage.value - 1) * 10
  return SCL90_QUESTIONS.slice(start, start + 10)
})

const getQuestionIndex = (idx) => (currentPage.value - 1) * 10 + idx

const submit = async () => {
  await store.submitTest(answers.value)
  router.push('/result')
}
</script>

<style scoped>
.test-view { padding: 20px; max-width: 800px; margin: 0 auto; }
.question-item { margin-bottom: 20px; padding: 15px; background: #f5f7fa; border-radius: 8px; }
.question-item p { margin-bottom: 10px; font-weight: 500; }
</style>
