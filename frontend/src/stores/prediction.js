import { defineStore } from 'pinia'
import client from '@/api/client'

export const usePredictionStore = defineStore('prediction', {
  state: () => ({
    result: null,
    batchResults: null,
    loading: false
  }),

  actions: {
    async submitTest(answers, epq = { E: 50, P: 50, N: 50, L: 50 }) {
      this.loading = true
      try {
        const result = await client.post('/api/predict/online', { answers, epq })
        this.result = result
        return result
      } finally {
        this.loading = false
      }
    },

    async submitManual(scl90, epq) {
      this.loading = true
      try {
        const result = await client.post('/api/predict/manual', { scl90, epq })
        this.result = result
        return result
      } finally {
        this.loading = false
      }
    },

    async uploadBatch(file) {
      this.loading = true
      try {
        const formData = new FormData()
        formData.append('file', file)
        const result = await client.post('/api/predict/batch', formData)
        this.batchResults = result
        return result
      } finally {
        this.loading = false
      }
    }
  }
})
