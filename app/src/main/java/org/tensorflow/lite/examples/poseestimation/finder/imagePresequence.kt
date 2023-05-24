package org.tensorflow.lite.examples.poseestimation.finder

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ExifInterface

object imagePresequence {


    fun exifOrientationToDegrees(exifOrientation: Int): Int {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) {
            return 90
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {
            return 180
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {
            return 270
        }
        return 0
    }


    fun rotate(bitmap: Bitmap?, degrees: Int): Bitmap? { // 이미지 회전 및 이미지 사이즈 압축
        var bitmap = bitmap
        if (degrees != 0 && bitmap != null) {
            val m = Matrix()
            m.setRotate(degrees.toFloat(), bitmap.width.toFloat() / 2,
                bitmap.height.toFloat() / 2)
            try {
                val converted = Bitmap.createBitmap(bitmap, 0, 0,
                    bitmap.width, bitmap.height, m, true)
                if (bitmap != converted) {
                    bitmap.recycle()
                    bitmap = converted

                    val height = bitmap.height
                    val width = bitmap.width

                    // 이미지 사이즈 줄이기
                    bitmap = Bitmap.createScaledBitmap(bitmap, 480, height / (width / 480), true)
                }
            } catch (ex: OutOfMemoryError) {
                // 메모리가 부족하여 회전을 시키지 못할 경우 그냥 원본을 반환합니다.
            }
        }
        return bitmap
    }
}