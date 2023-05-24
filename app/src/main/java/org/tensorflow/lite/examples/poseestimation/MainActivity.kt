/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.*
import org.tensorflow.lite.examples.poseestimation.finder.imagePresequence
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date


class MainActivity : AppCompatActivity() {
    // Manifest 에서 설정한 권한을 가지고 온다.
    val CAMERA_PERMISSION = arrayOf(Manifest.permission.CAMERA)
    val STORAGE_PERMISSION = arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE)

    // 권한 플래그값 정의
    val FLAG_PERM_CAMERA = 98
    val FLAG_PERM_STORAGE = 99

    // 카메라와 갤러리를 호출하는 플래그
    val FLAG_REQ_CAMERA = 101
    val FLAG_REA_STORAGE = 102

    //xml 개체 호출
    private lateinit var buttonListener: Button
    lateinit var imgViewer: ImageView

    // 사진을 저장할 위치
    lateinit var currentPhotoPath: String

    /** A [SurfaceView] for camera preview.   */
    private lateinit var surfaceView: SurfaceView

    /** Default pose estimation model is 1 (MoveNet Thunder) **/
    private var modelPos = 1

    /** Default device is CPU */
    private var device = Device.CPU
    private lateinit var spnDevice: Spinner
    private lateinit var spnModel: Spinner
    private lateinit var spnTracker: Spinner
    private lateinit var vTrackerOption: View
    private lateinit var swClassification: SwitchCompat
    private lateinit var vClassificationOption: View
    private var cameraSource: CameraSource? = null
    private var isClassifyPose = false
    private var changeModelListener = object : AdapterView.OnItemSelectedListener {
        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }

        override fun onItemSelected(
            parent: AdapterView<*>?,
            view: View?,
            position: Int,
            id: Long
        ) {
            changeModel(position)
        }
    }
    private var changeDeviceListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            // Do nothing
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }
    }
    private var changeTrackerListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            // Do nothing
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // keep screen on while app is running
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        spnModel = findViewById(R.id.spnModel)
        spnDevice = findViewById(R.id.spnDevice)
        spnTracker = findViewById(R.id.spnTracker)
        vTrackerOption = findViewById(R.id.vTrackerOption)
        surfaceView = findViewById(R.id.surfaceView)
        swClassification = findViewById(R.id.swPoseClassification)
        vClassificationOption = findViewById(R.id.vClassificationOption)
        initSpinner()
        spnModel.setSelection(modelPos)

        // xml 개체 초기화
        imgViewer = findViewById(R.id.img_viewer)
        buttonListener = findViewById(R.id.pic_btn)

        // 화면이 만들어 지면서 저장소 권한을 체크 합니다.
        // 권한이 승인되어 있으면 카메라를 호출하는 메소드를 실행합니다.
        setViews()

        if (cameraSource == null) {
            cameraSource =
                CameraSource(surfaceView, object : CameraSource.CameraSourceListener {
                    override fun onFPSListener(fps: Int) {
                        // Do nothing
                    }

                    override fun onDetectedInfo(
                        personScore: Float?,
                        poseLabels: List<Pair<String, Float>>?
                    ) {
                        // Do nothing
                    }

                }).apply {
                    prepareCamera()
                }
            isPoseClassifier()
            lifecycleScope.launch(Dispatchers.Main) {
                // Do nothing
            }
        }
    }

    private fun setViews() {
        //카메라 버튼 클릭
        if(checkPermission(CAMERA_PERMISSION,FLAG_PERM_CAMERA)){
            buttonListener.setOnClickListener {
                val builder = AlertDialog.Builder(this)
                builder
                    .setTitle("Title")
                    .setMessage("MessageMessageMessageMessageMessageMessage")
                    .setPositiveButton("Start",
                        DialogInterface.OnClickListener { dialog, id ->
                            // Start 버튼 선택시 수행
                            openCamera()
                        })
                builder.create()
                builder.show()
                //카메라 호출 메소드
            }
        }
    }

    // 권한이 있는지 체크하는 메소드
    private fun checkPermission(permissions:Array<out String>,flag:Int):Boolean{
        for(permission in permissions){
            // 만약 권한이 승인되어 있지 않다면 권한승인 요청을 사용에 화면에 호출합니다.
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED)
                ActivityCompat.requestPermissions(this, permissions, flag)
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED)
                return false
        }
        return true
    }

    // 사진을 저장해주는 메소드
    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    // open camera
    private fun openCamera() {

        createPoseEstimator()

        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)

        // Ensure that there's a camera activity to handle the intent
        intent.resolveActivity(packageManager)?.also {
            // Create the File where the photo should go
            val photoFile: File? = try {
                createImageFile()
            } catch (ex: IOException) {
                // Error occurred while creating the File
                null
            }
            // Continue only if the File was successfully created
            photoFile?.also {
                val photoURI: Uri = FileProvider.getUriForFile(
                    this,
                    "com.example.android.fileprovider",
                    it
                )
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)

                startActivityForResult(intent, FLAG_REQ_CAMERA)
            }
        }
    }

    private fun isPoseClassifier() {
        cameraSource?.setClassifier(if (isClassifyPose) PoseClassifier.create(this) else null)
    }

    // Initialize spinners to let user select model/accelerator/tracker.
    private fun initSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_models_array,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            // Specify the layout to use when the list of choices appears
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            // Apply the adapter to the spinner
            spnModel.adapter = adapter
            spnModel.onItemSelectedListener = changeModelListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_device_name, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnDevice.adapter = adaper
            spnDevice.onItemSelectedListener = changeDeviceListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_tracker_array, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnTracker.adapter = adaper
            spnTracker.onItemSelectedListener = changeTrackerListener
        }
    }

    // Change model when app is running
    private fun changeModel(position: Int) {
        if (modelPos == position) return
        modelPos = 1
        createPoseEstimator()
    }

    private fun createPoseEstimator() {
        // For MoveNet MultiPose, hide score and disable pose classifier as the model returns
        // multiple Person instances.
        val poseDetector = when (modelPos) {
            1 -> {
                // MoveNet Thunder (SinglePose)
                spnTracker.setSelection(0)
                MoveNet.create(this, device, ModelType.Thunder)
            }
            else -> {
                null
            }
        }
        poseDetector?.let { detector ->
            cameraSource?.setDetector(detector)
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    /**
     * Shows an error message dialog.
     */
    class ErrorDialog : DialogFragment() {
        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(requireArguments().getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    // do nothing
                }
                .create()

        companion object {

            @JvmStatic
            private val ARG_MESSAGE = "message"

            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }

    // checkPermission() 에서 ActivityCompat.requestPermissions을 호출한 다음 사용자가 권한 허용 여부를 선택하면 해당 메소드로 값이 전달됩니다.
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when(requestCode){
            FLAG_PERM_STORAGE ->{
                for(grant in grantResults){
                    if(grant != PackageManager.PERMISSION_GRANTED){
                        // 권한이 승인되지 않았다면 return을 사용하여 메소드를 종료시켜 줍니다
                        Toast.makeText(this,"저장소 권한을 승인해야지만 앱을 사용할 수 있습니다..",Toast.LENGTH_SHORT).show()
                        finish()
                        return
                    }
                }
                // 카메라 호출 메소드
                setViews()
            }
            FLAG_PERM_CAMERA ->{
                for(grant in grantResults){
                    if(grant != PackageManager.PERMISSION_GRANTED){
                        Toast.makeText(this,"카메라 권한을 승인해야지만 카메라를 사용할 수 있습니다.",Toast.LENGTH_SHORT).show()
                        return
                    }
                }
                openCamera()
            }
        }
    }


    // startActivityForResult 을 사용한 다음 돌아오는 결과값을 해당 메소드로 호출합니다.
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(resultCode == RESULT_OK){
            when(requestCode){
                FLAG_REQ_CAMERA ->{
                    val file = File(currentPhotoPath)
                    var cacheBitmap = MediaStore.Images.Media.getBitmap(contentResolver, Uri.fromFile(file))
                    if (cacheBitmap != null) {
                        val exif = currentPhotoPath?.let { ExifInterface(it) }
                        val exifOrientation: Int = exif!!.getAttributeInt(
                            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
                        val exifDegree = imagePresequence.exifOrientationToDegrees(exifOrientation)
                        cacheBitmap = imagePresequence.rotate(cacheBitmap, exifDegree)
                    }
                    else Log.d("myBitmap null", "null")

                    cacheBitmap = cameraSource?.initCamera(cacheBitmap)
                    imgViewer.setImageBitmap(cacheBitmap)

                }
            }
        }
    }
}