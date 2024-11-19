// Generated by gencpp from file fast_livo/States.msg
// DO NOT EDIT!


#ifndef FAST_LIVO_MESSAGE_STATES_H
#define FAST_LIVO_MESSAGE_STATES_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace fast_livo
{
template <class ContainerAllocator>
struct States_
{
  typedef States_<ContainerAllocator> Type;

  States_()
    : header()
    , rot_end()
    , pos_end()
    , vel_end()
    , bias_gyr()
    , bias_acc()
    , gravity()
    , cov()  {
    }
  States_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , rot_end(_alloc)
    , pos_end(_alloc)
    , vel_end(_alloc)
    , bias_gyr(_alloc)
    , bias_acc(_alloc)
    , gravity(_alloc)
    , cov(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _rot_end_type;
  _rot_end_type rot_end;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _pos_end_type;
  _pos_end_type pos_end;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _vel_end_type;
  _vel_end_type vel_end;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _bias_gyr_type;
  _bias_gyr_type bias_gyr;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _bias_acc_type;
  _bias_acc_type bias_acc;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _gravity_type;
  _gravity_type gravity;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _cov_type;
  _cov_type cov;





  typedef boost::shared_ptr< ::fast_livo::States_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::fast_livo::States_<ContainerAllocator> const> ConstPtr;

}; // struct States_

typedef ::fast_livo::States_<std::allocator<void> > States;

typedef boost::shared_ptr< ::fast_livo::States > StatesPtr;
typedef boost::shared_ptr< ::fast_livo::States const> StatesConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::fast_livo::States_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::fast_livo::States_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::fast_livo::States_<ContainerAllocator1> & lhs, const ::fast_livo::States_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.rot_end == rhs.rot_end &&
    lhs.pos_end == rhs.pos_end &&
    lhs.vel_end == rhs.vel_end &&
    lhs.bias_gyr == rhs.bias_gyr &&
    lhs.bias_acc == rhs.bias_acc &&
    lhs.gravity == rhs.gravity &&
    lhs.cov == rhs.cov;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::fast_livo::States_<ContainerAllocator1> & lhs, const ::fast_livo::States_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace fast_livo

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::fast_livo::States_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::fast_livo::States_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::fast_livo::States_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::fast_livo::States_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::fast_livo::States_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::fast_livo::States_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::fast_livo::States_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4a896a0d8c07506c836e98c3fa512a5e";
  }

  static const char* value(const ::fast_livo::States_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4a896a0d8c07506cULL;
  static const uint64_t static_value2 = 0x836e98c3fa512a5eULL;
};

template<class ContainerAllocator>
struct DataType< ::fast_livo::States_<ContainerAllocator> >
{
  static const char* value()
  {
    return "fast_livo/States";
  }

  static const char* value(const ::fast_livo::States_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::fast_livo::States_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header          # timestamp of the first lidar in a frame\n"
"float64[] rot_end      # the estimated attitude (rotation matrix) at the end lidar point\n"
"float64[] pos_end      # the estimated position at the end lidar point (world frame)\n"
"float64[] vel_end      # the estimated velocity at the end lidar point (world frame)\n"
"float64[] bias_gyr     # gyroscope bias\n"
"float64[] bias_acc     # accelerator bias\n"
"float64[] gravity      # the estimated gravity acceleration\n"
"float64[] cov          # states covariance\n"
"# Pose6D[] IMUpose        # 6D pose at each imu measurements\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::fast_livo::States_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::fast_livo::States_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.rot_end);
      stream.next(m.pos_end);
      stream.next(m.vel_end);
      stream.next(m.bias_gyr);
      stream.next(m.bias_acc);
      stream.next(m.gravity);
      stream.next(m.cov);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct States_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::fast_livo::States_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::fast_livo::States_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "rot_end[]" << std::endl;
    for (size_t i = 0; i < v.rot_end.size(); ++i)
    {
      s << indent << "  rot_end[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.rot_end[i]);
    }
    s << indent << "pos_end[]" << std::endl;
    for (size_t i = 0; i < v.pos_end.size(); ++i)
    {
      s << indent << "  pos_end[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.pos_end[i]);
    }
    s << indent << "vel_end[]" << std::endl;
    for (size_t i = 0; i < v.vel_end.size(); ++i)
    {
      s << indent << "  vel_end[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.vel_end[i]);
    }
    s << indent << "bias_gyr[]" << std::endl;
    for (size_t i = 0; i < v.bias_gyr.size(); ++i)
    {
      s << indent << "  bias_gyr[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.bias_gyr[i]);
    }
    s << indent << "bias_acc[]" << std::endl;
    for (size_t i = 0; i < v.bias_acc.size(); ++i)
    {
      s << indent << "  bias_acc[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.bias_acc[i]);
    }
    s << indent << "gravity[]" << std::endl;
    for (size_t i = 0; i < v.gravity.size(); ++i)
    {
      s << indent << "  gravity[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.gravity[i]);
    }
    s << indent << "cov[]" << std::endl;
    for (size_t i = 0; i < v.cov.size(); ++i)
    {
      s << indent << "  cov[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.cov[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // FAST_LIVO_MESSAGE_STATES_H
