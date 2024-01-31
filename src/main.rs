use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};

fn main() {
    // Setup
    let instance = setup_instance();
    let physical_device = get_physical_device(instance);
    let (device, queues) = get_logical_device(physical_device);
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

    let (device, mut queues) = Device::new(
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
