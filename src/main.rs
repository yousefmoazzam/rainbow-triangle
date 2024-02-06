use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::format::Format;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo,
    PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    SubpassEndInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::pipeline::{GraphicsPipeline, PipelineShaderStageCreateInfo, PipelineLayout};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::color_blend::{ColorBlendState, ColorBlendAttachmentState};
use vulkano::sync::{self, GpuFuture};

use image::{ImageBuffer, Rgba};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    colour: [f32; 3],
}

fn main() {
    // Setup
    let instance = setup_instance();
    let physical_device = get_physical_device(instance);
    let (device, queues) = get_logical_device(physical_device);
    let queue = get_queue(queues);
    let memory_allocator = create_memory_allocator(device.clone());

    // Image creation
    let height = 1024;
    let width = 1024;
    let image = create_image(memory_allocator.clone(), height, width);

    // Create vertices of single triangle
    let vertex1 = MyVertex { position: [0.0, -0.5], colour: [1.0, 0.0, 0.0] };
    let vertex2 = MyVertex { position: [0.5, 0.5], colour: [0.0, 1.0, 0.0] };
    let vertex3 = MyVertex { position: [-0.5, 0.5], colour: [0.0, 0.0, 1.0] };

    // Create buffer on host to hold vertex data purely for transferring to "device-local" GPU
    // memory (called a "staging buffer")
    let staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    ).expect("Should have been able to create staging buffer");

    // Create vertex buffer in "device-local" memory which will be the copy destination of the
    // three triangle vertices
    let vertex_buffer = Buffer::new_slice::<MyVertex>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        3,
    ).expect("Should have been able to create vertex buffer");

    // Create command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Copy vertex data from staging buffer to device-local buffer
    copy_from_staging_to_device(
        device.clone(),
        queue.clone(),
        &command_buffer_allocator,
        staging_buffer.clone(),
        vertex_buffer.clone(),
    );

    // Create render pass object configured to clear a single image
    let render_pass = create_render_pass(device.clone());

    // Create framebuffer that contains the single image
    let framebuffer = wrap_image_in_framebuffer(image.clone(), render_pass.clone());

    // Create command buffer builder
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("Should have been able to create command buffer builder");

    // Create shader module objects
    let vertex_shader = vertex_shaders::load(device.clone())
        .expect("Should be able to create shader module for vertex shader");
    let fragment_shader = fragment_shaders::load(device.clone())
        .expect("Should be able to create shader module for fragment shader");

    // Create viewport
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [height as f32, width as f32],
        depth_range: 0.0..=1.0,
    };

    // Create graphics pipeline object
    let graphics_pipeline = create_graphics_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        render_pass.clone(),
        viewport.into(),
    );

    // Create buffer to store the drawn image on the CPU/host
    let host_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..height * width * 4).map(|_| 0u8),
    ).expect("Should be able to create buffer on host to hold drawn image");

    // Record commands to builder
    let colour: [f32; 4] = [0.80, 0.97, 1.00, 1.00];
    configure_command_buffer_builder(
        &mut builder,
        colour,
        framebuffer,
        vertex_buffer.clone(),
        graphics_pipeline.clone(),
        image.clone(),
        host_buffer.clone(),
    );

    // Create command buffer object from builder
    let command_buffer = builder.build()
        .expect("Should be able to create command buffer out of builder object");

    // Create future object to manage/oversee execution of pipeline
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .expect("Should be able to submit commands in command buffer for execution")
        .then_signal_fence_and_flush()
        .expect("Should be able to ask for fence that signals execution completion on GPU");

    // Wait for "fence"/signal from GPU and block CPU until it's received
    future.wait(None).expect("Should be able to wait for 'fence' from GPU");

    // Save image in png file
    let buffer_content = host_buffer.read()
        .expect("Should be able to read contents of host buffer");
    let drawn_image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, &buffer_content[..])
        .expect("Should be able to create an image object from the host buffer");
    drawn_image.save("triangle.png").expect("Unable to save png image");
}

fn setup_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No Vulkan library installed");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("Failed to create Vulkan instance");
    instance
}

fn get_physical_device(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
    let mut physical_devices = instance.enumerate_physical_devices()
        .expect("Couldn't find any physical devices");
    let physical_device = physical_devices.next().expect("No device found");
    physical_device
}

fn get_logical_device(
    physical_device: Arc<PhysicalDevice>,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;

    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![
                QueueCreateInfo { queue_family_index, ..Default::default() },
            ],
            ..Default::default()
        },
    ).expect("Unable to create logical device from physical device");
    (device, queues)
}

fn get_queue(mut queues: impl ExactSizeIterator<Item = Arc<Queue>>) -> Arc<Queue> {
    let queue = queues.next().expect("Should have had 1 queue in the iterator of queues");
    queue
}

fn create_memory_allocator(device: Arc<Device>) -> Arc<StandardMemoryAllocator> {
    let allocator = Arc::new(
        StandardMemoryAllocator::new_default(device)
    );
    allocator
}

fn create_image(
    allocator: Arc<StandardMemoryAllocator>,
    height: u32,
    width: u32,
) -> Arc<Image> {
    let image = Image::new(
        allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [height, width, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).expect("Should've been able to create a single 2D image");
    image
}

fn create_render_pass(device: Arc<Device>) -> Arc<RenderPass> {
    let render_pass = vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: Format::R8G8B8A8_SRGB,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    ).expect("Should have been able to create render pass");
    render_pass
}

fn wrap_image_in_framebuffer(
    image: Arc<Image>,
    render_pass: Arc<RenderPass>,
) -> Arc<Framebuffer> {
    let view = ImageView::new_default(image)
        .expect("Should have been able to create an image view");
    let framebuffer = Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    ).expect("Should have been able to create framebuffer");
    framebuffer
}

fn configure_command_buffer_builder(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    colour: [f32; 4],
    framebuffer: Arc<Framebuffer>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    input_image: Arc<Image>,
    output_buffer: Subbuffer<[u8]>,
) {
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(colour.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("Should be able to configure image clearing in render pass")
        .bind_pipeline_graphics(graphics_pipeline)
        .expect("Should be able to bind graphics pipeline object")
        .bind_vertex_buffers(0, vertex_buffer)
        .expect("Should be able to bind single vertex buffer")
        .draw(3, 1, 0, 0)
        .expect("Should be able to draw vertices")
        .end_render_pass(SubpassEndInfo::default())
        .expect("Should be able to configure end of render pass")
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(input_image, output_buffer))
        .expect("Should be able to copy drawn image to output buffer");
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vertex_shader.entry_point("main")
        .expect("Expected main() function in vertex shader");
    let fs = fragment_shader.entry_point("main")
        .expect("Expected main() function in fragment shader");

    // TODO: Learn more, not much is said in vulkano triangle tutorial about this
    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .expect("Should be able to get vertex input state from vertex struct");

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // TODO: Learn more, not much is said in vulkano triangle tutorial about this
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone())
        .expect("Should be able to create pipeline layout info")
    ).expect("Should be able to create pipeline layout");

    let subpass = Subpass::from(render_pass, 0)
        .expect("Should be able to create single subpass");

    let graphics_pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            // Need the `into_iter().collect()` to get a `smallvec::SmallVec`, not sure why can't
            // create directly though
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                // Need the `into_iter().collect()` to get a `smallvec::SmallVec`, not sure why
                // can't create directly though
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        },
    ).expect("Should be able to create graphics pipeline object");

    graphics_pipeline
}

fn copy_from_staging_to_device<T>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    allocator: &StandardCommandBufferAllocator,
    staging_buffer: Subbuffer<[T]>,
    device_buffer: Subbuffer<[T]>,
) {
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("Should be able to create command buffer for copy command");

    builder
        .copy_buffer(CopyBufferInfo::buffers(staging_buffer, device_buffer))
        .expect("Should be able to record copy command");

    let command_buffer = builder.build()
        .expect("Should be able to build command buffer for copying");

    let future = sync::now(device.clone())
        .then_execute(queue, command_buffer)
        .expect("Should be able to submit command buffer w/ copy command")
        .then_signal_fence_and_flush()
        .expect("Should be able to ask for fence for when copy command has completed");

    future.wait(None).expect("Should be able to wait for fence from future");
}

mod vertex_shaders {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec3 colour;

            layout(location = 0) out vec3 fragColour;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                fragColour = colour;
            }
        ",
    }
}

mod fragment_shaders {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec3 fragColour;
            layout(location = 0) out vec4 outColour;

            void main() {
                outColour = vec4(fragColour, 1.0);
            }
        ",
    }
}
