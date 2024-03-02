package com.example.landmark_recognition.data

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProcessor
import androidx.compose.material3.Surface
import com.example.landmark_recognition.domain.Classificatiion
import com.example.landmark_recognition.domain.LandmarkClassifier
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.lang.IllegalStateException

class TfLiteLandmarkClassifier(
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 1 // or 2/3 for multiple possilbe results
): LandmarkClassifier {
    private var classifier: ImageClassifier? = null
    private fun setupClassifier(){
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()
        val options = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(maxResults)
            .setScoreThreshold(threshold)
            .build()

        try {
            classifier = ImageClassifier.createFromFileAndOptions(
                context,
                "landmarks.tflite",
                options
            )
        }catch (e: IllegalStateException){
            e.printStackTrace()
        }
    }
    override fun classify(bitmap: Bitmap, rotation: Int): List<Classificatiion> {
        if (classifier == null){
            setupClassifier()
        }
        val imageProcessor = org.tensorflow.lite.support.image.ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()
        val results = classifier?.classify(tensorImage, imageProcessingOptions)
        return results?.flatMap { classification->
            classification.categories.map { category ->
                Classificatiion(
                    name = category.displayName,
                    score = category.score
                        )
            }
        }?.distinctBy { it.name } ?: emptyList()
    }
    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation{
        return when(rotation){
            android.view.Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            android.view.Surface.ROTATION_90 -> ImageProcessingOptions.Orientation.TOP_LEFT
            android.view.Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP

        }
    }

}